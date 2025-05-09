import os
import re
import xarray as xr
import pandas as pd
import numpy as np
import datetime as dt
import parse
import logging
from unitpy import Unit
from toolz import keyfilter, valfilter, assoc_in
import trosat.sunpos as sp

import taro
import taro.utils
import taro.futils
import taro.qcrad

logger = logging.getLogger(__name__)

def _parse_unit(unit):
    unit = str(unit)
    if unit == "%":
        return ''
    if unit == "nan":
        return ''
    # add ^ before numbers
    unit = re.sub(r"-?[0-9]",r"^\g<0>", unit)
    return unit

def _parse_quantity(unit):
    punit = _parse_unit(unit)
    if unit == "%":
        return 1e-2*Unit(punit)
    return 1.*Unit(punit)

def to_l1a(
        fname: str,
        *,
        config=None,
        global_attrs=None):
    """
    Read logger raw file and parse it to xarray Dataset. Thereby, attributes and names are defined via cfmeta.json file and sun position values are calculated and added.

    Parameters
    ----------
    fname: str
        Path and filename of the raw logger file.
    config: dict
        Stores processing specific configuration.
            * cfjson -> path to cfmeta.json, the default is "../share/pyrnet_cfmeta.json"
            * stripminutes -> number of minutes to be stripped from the data at start and end,
                the default is 5.
    global_attrs: dict
        Additional global attributes for the Dataset. (Overrides cfmeta.json attributes)
    Returns
    -------
    xarray.Dataset
        Raw Logger data for one measurement periode.
    """
    config = taro.utils.merge_config(config)
    gattrs, vattrs, vencode = taro.futils.get_cfmeta(config)

    if global_attrs is not None:
        gattrs.update(global_attrs)

    # 1. Parse raw file
    # Parse table info
    fname_info = parse.parse(
        config["fname_out"],
        os.path.basename(fname)
    )
    table_config = taro.utils.read_json(config["file_logger_tables"])[fname_info["table"]]
    usecols = np.argwhere([v is not None for v in table_config]).ravel()

    table_map = np.array([table_config[i] for i in usecols])

    names_with_sfx = [(v+'_'+s).strip('_') for v,s in zip(table_map[:,0],table_map[:,1])]

    names = []
    for i, v in enumerate(names_with_sfx):
        count = list(table_map[:i, 0]).count(v)
        names.append(v + f"_{count+1:d}" if count > 0 else v)

    # parse the data file
    dat = pd.read_csv(
        fname,
        skiprows=4,
        usecols=usecols,
        names=names,
        header=None,
        sep=','
    )

    dat["time"] = pd.to_datetime(dat['time'].astype(str), format='mixed')
    ds = dat.set_index("time").to_xarray()
    ds = ds.drop_vars(["record"])
    ds = ds.astype(float)  # ensure "NaN" to np.nan

    # add meta
    for name, troposid in zip(names,table_map[:,2]):
        if (name == "record") or (name == "time"):
            continue
        meta = taro.utils.meta_lookup(config,troposID=troposid)
        # drop calibration meta
        meta = keyfilter(lambda x: not x.startswith('calibration'), meta)
        # drop None values
        meta = valfilter(lambda x: x is not None, meta)
        if "troposID" in meta:
            meta["ID"] = meta.pop("troposID")
        # update netcdf attributes
        ds[name].attrs.update(meta)

    ## Scale variables to correct units
    # identify logger units
    colunits = pd.read_csv(
        fname,
        skiprows=2,
        nrows=1,
        usecols=usecols,
        names=names,
        header=None,
        sep=','
    )
    colunits = colunits.set_index("time").to_xarray()
    colunits = colunits.drop_vars(["record"])

    # convert to cfmeta units
    for i,key in enumerate(names):
        if (key == "record") or (key == "time"):
            continue
        oldunit = colunits[key].values[0]
        newunit = vattrs[table_map[i, 0]]["units"]
        if oldunit == "%" and newunit == "1":
            ds[key].values = ds[key].values*1e-2
            continue

        oldunit = _parse_unit(oldunit)
        newunit = _parse_unit(newunit)

        oldval = ds[key].values * Unit(oldunit)
        ds[key].values = (oldval.to(newunit)).value

    # add coordinats if available in config
    if config["coordinates"] is not None:
        lat, lon, alt = config["coordinates"]
        if lat is not None:
            ds["lat"] = lat
        if lon is not None:
            ds["lon"] = lon
        if alt is not None:
            ds["altitude"] = alt


    # 2. Add global meta data
    now = pd.to_datetime(np.datetime64("now"))
    gattrs.update({
        'processing_level': 'l1a',
        'product_version': taro.__version__,
        'history': f'{now.isoformat()}: Generated level l1a  by taro version {taro.__version__}; ',
    })
    ds.attrs.update(gattrs)

    # drop occurrence of duplicate sample values
    ds = ds.drop_duplicates("time")

    # add global coverage attributes
    ds = taro.futils.update_coverage_meta(ds, timevar="time")

    # add attributes to Dataset
    for k,v in vattrs.items():
        # iterate over suffixed variables
        for ki in [key for key in ds if key.startswith(k)]:
            ds[ki].attrs.update(v)

    # add encoding to Dataset
    ds = taro.futils.add_encoding(ds, vencode)

    return ds


def to_l1b(ds_l1a, resolution, *, config=None):
    config = taro.utils.merge_config(config)
    gattrs, vattrs, vencode = taro.futils.get_cfmeta(config)

    if ds_l1a.processing_level != "l1a":
        logger.warning(f"Is not a l1a file. Skip.")
        return None

    flx_vars = config["l1b_flux_variables"]
    ds_l1b = ds_l1a.copy()

    # 3. add coordinats if available in config
    if config["coordinates"] is not None:
        lat, lon, alt = config["coordinates"]
        if lat is not None:
            ds_l1b["lat"] = lat
        if lon is not None:
            ds_l1b["lon"] = lon
        if alt is not None:
            ds_l1b["altitude"] = alt

    # 4. add sun position
    if ("lat" in ds_l1b) and ("lon" in ds_l1b):
        szen, sazi = sp.sun_angles(
            time=ds_l1b.time.values,
            lat=ds_l1b.lat.values,
            lon=ds_l1b.lon.values
        )
        szen = szen.squeeze()
        sazi = sazi.squeeze()

        esd = np.mean(sp.earth_sun_distance(ds_l1b.time.values))

        ds_l1b = ds_l1b.assign(
            {
                "szen": (("time"), szen),
                "sazi": (("time"), sazi),
                "esd":  esd
            }
        )
        # update attributes and encoding
        for key in ['szen', 'sazi', 'esd']:
            ds_l1b[key].attrs.update(vattrs[key])

    # # x. correct dark current
    # for var in flx_vars:
    #     var = f"raw_{var}" # apply to raw data only
    #     if "longwave" in ds_l1b[var].attrs["standard_name"]:
    #         continue
    #     dark = np.nanmedian(ds_l1b[var].values[szen>100])
    #     ds_l1b[var].values = ds_l1b[var].values - dark
    #     ds_l1b[var].attrs.update({
    #         "dark_offset": dark,
    #         "dark_offset_units":ds_l1b[var].attrs['units']
    #     })

    # 1. calibrate flux vars
    ds_l1b = ds_l1b.drop_vars(flx_vars, errors="ignore") # drop any pre-calibrated variables which share the flux variable name 
    for var in flx_vars:
        var = f"raw_{var}" # apply to raw data only
        troposID = ds_l1b[var].attrs["ID"]
        calib = taro.utils.parse_calibration(
            cfile=config["file_calibration"],
            troposID=troposID,
            cdate=ds_l1b.time.values[0]
        )

        cfac_unit = _parse_quantity(calib["calibration_factor_units"].values)
        cfac = calib["calibration_factor"].values * cfac_unit
        # assume cfac unit in the form ( units_of_Voltage / (Wm-2) )
        cfac = ((1./cfac).to(f"W m^-2 {ds_l1b[var].attrs['units']}^-1")).value
        cdate = f"{pd.to_datetime(calib.time.values):%Y-%m}"
        
        # calibration factor temperature sensitivity correction
        # select sensor temperature
        dstemp = ds_l1b.filter_by_attrs(ID=troposID)
        tcorr_flag = False
        if len(dstemp)!=0:
            for key in dstemp:
                if key.startswith("sensor_temperature"):
                    temp_sensor = dstemp[key].values * Unit(dstemp[key].attrs['units'])
            t_sensor = temp_sensor.to("degC").value
            coeff_a = (calib["temperature_coeff_a"].values * _parse_quantity(calib["temperature_coeff_a_units"].values)).to("degC^-2").value
            coeff_b = (calib["temperature_coeff_b"].values * _parse_quantity(calib["temperature_coeff_b_units"].values)).to("degC^-1").value
            coeff_c = calib["temperature_coeff_c"].values
            cfac = np.full(t_sensor.size,cfac) / (coeff_a*t_sensor**2 + coeff_b*t_sensor + coeff_c)
            tcorr_flag = True

        # Longwave calibration
        if "longwave" in ds_l1b[var].attrs["standard_name"]:
            # select sensor temperature
            dstemp = ds_l1b.filter_by_attrs(ID=troposID)
            for key in dstemp:
                if key.startswith("sensor_temperature"):
                    temp_sensor = dstemp[key].values * Unit(dstemp[key].attrs['units'])
            # measured Voltage
            V0 = ds_l1b[var].values
            # calibrate to W m-2
            ds_l1b[var].values = V0 * cfac + 5.670367e-8 * ((temp_sensor.to("K")).value**4)
            if tcorr_flag:
                ds_l1b[var].attrs.update({
                    "calibration_function": "flux (W m-2) = flux (V) * calibration_factor/temperatur_correction (W m-2 V-1) + 5.670367e-8 * (sensor_temperature (K))**4",
                    "temperatur_correction": "temperatur_correction = coeff[0] * sensor_temperature(degC)**2 + coeff[1] * sensor_temperature(degC) + coeff[2]",
                    "temperatur_correction_coeff": [coeff_a, coeff_b, coeff_c]
                })
            else:
                ds_l1b[var].attrs.update({
                    "calibration_function": "flux (W m-2) = flux (V) * calibration_factor (W m-2 V-1) + 5.670367e-8 * (sensor_temperature (K))**4",
                })
        else:
            ds_l1b[var].values = ds_l1b[var].values*cfac
            # ds_l1b[var].values[ds_l1b[var].values < 0] = 0.
            if tcorr_flag:
                ds_l1b[var].attrs.update({
                    "calibration_function": "flux (W m-2) = flux (V) * calibration_factor/temperatur_correction (W m-2 V-1)",
                    "temperatur_correction": "temperatur_correction = coeff[0] * sensor_temperature(degC)**2 + coeff[1] * sensor_temperature(degC) + coeff[2]",
                    "temperatur_correction_coeff": [coeff_a, coeff_b, coeff_c]
                })
            else:
                ds_l1b[var].attrs.update({
                    "calibration_function": "flux (W m-2) = flux (V) * calibration_factor (W m-2 V-1)",
                })
        # add new attributes and encoding to calibrated flux vars
        ds_l1b[var].attrs.update({
            "units": "W m-2",
            "calibration_factor": calib["calibration_factor"].values,
            "calibration_factor_units": calib["calibration_factor_units"].values,
            "calibration_error": calib["calibration_error"].values,
            "calibration_error_units": calib["calibration_error_units"].values,
            "calibration_date": cdate,
        })
        scale_factor = 1e-4
        add_offset = -10.
        valid_range = np.array([0, 2000])  # valid range in data units W m-2
        valid_range = ((valid_range - add_offset)/scale_factor).astype(int)
        vencode = assoc_in(vencode, [var,'valid_range'],list(valid_range))
        vencode = assoc_in(vencode, [var, 'scale_factor'], scale_factor)
        vencode = assoc_in(vencode, [var, 'add_offset'], add_offset)
    ds_l1b = ds_l1b.rename_vars({"raw_"+v:v for v in flx_vars}) # drop "_raw" suffix for calibrated variables

    # 2. resample dataset
    methods = ['mean'] + config["l1b_resample_stats"]
    res = taro.futils.resample(
        ds_l1b,
        freq=resolution,
        methods=methods,
        kwargs=dict(skipna=True)
    )
    ds_l1b = res[0]
    for i, method in enumerate(methods[1:]):
        for var in flx_vars:
            ds_l1b[f"{var}_{method}"] = res[i+1][var]
            ds_l1b[f"{var}_{method}"].attrs.update({
                "standard_name": f"{method}_"+ds_l1b[f"{var}_{method}"].attrs["standard_name"]
            })

    # Correct Wind direction with sensor compass
    if "compass" in ds_l1b:
        for key in ds_l1b:
            if key.startswith("wind_direction") and not (key.endswith("min") or key.endswith("std")):
                tvals = ds_l1b[key].values
                tvals += ds_l1b["compass"].values
                tvals[tvals>360] -= 360.
                ds_l1b[key].values = tvals
                ds_l1b[key].attrs.update({"comment": "Corrected with sensor compass."})

    # 5. add BSRN quality flags
    ds_l1b = taro.qcrad.quality_control(ds_l1b)

    # 6. add global coverage attributes
    ds_l1b = taro.futils.update_coverage_meta(ds_l1b, timevar="time")
    ds_l1b.attrs["processing_level"] = 'l1b'
    now = pd.to_datetime(np.datetime64("now"))
    ds_l1b.attrs["history"] = ds_l1b.history + f"{now.isoformat()}: Generated level l1b  by taro version {taro.__version__}; "
    ds_l1b.attrs['product_version'] = taro.__version__

    # update encoding
    ds_l1b = taro.futils.add_encoding(ds_l1b, vencode=vencode)

    return ds_l1b


def wiser_to_l1a(date, pf, *, config=None, global_attrs=None):
    # load config and nc config
    config = taro.utils.merge_config(config)
    gattrs, vattrs, vencode = taro.futils.get_cfmeta(config)

    if global_attrs is not None:
        gattrs.update(global_attrs)

    # load raw data
    date = pd.to_datetime(date)
    datetime = dt.datetime(date.year, date.month, date.day, 0, 0, 0)
    fname = os.path.join(pf, config['wiser_raw'])

    new = True
    for hour in np.arange(24):
        try:
            df = pd.read_csv(fname.format(
                dt=datetime + dt.timedelta(hours=int(hour)),
                campaign=config["campaign"],
                sfx='CSV'
            ))
        except Exception as error:
            # handle the exception
            print("An exception occurred:", error)
            logger.error("An exception occurred: " + str(error))
            continue

        df.pop(df.columns[-1])
        try:
            time = (f"{datetime:%Y-%m-%d}T" + df.head(1).astype(str)).values[0, 1::3].astype("datetime64[ns]")
        except Exception as error:
            # handle the exception
            print("An exception occurred:", error)
            logger.error("An exception occurred: " + str(error))
            continue
        wvls = df.values[7:, 0].astype(float)  # [nm]
        values_711 = (df.values[7:, 1::3].astype(float).T) * 1e-3  # [W m-2 nm-1]
        values_713 = (df.values[7:, 2::3].astype(float).T) * 1e-3  # [W m-2 nm-1]
        values_merge = (df.values[7:, 3::3].astype(float).T) * 1e-3  # [W m-2 nm-1]
        exposure_711 = df.values[3, 1::3].astype(float)  # [ms]
        exposure_713 = df.values[3, 2::3].astype(float)  # [ms]
        temp_711 = df.values[4, 1::3].astype(float)  # [degC]
        temp_713 = df.values[4, 2::3].astype(float)  # [degC]
        pwr_711 = df.values[5, 1::3].astype(float)  # [V]
        pwr_713 = df.values[5, 2::3].astype(float)  # [V]
        dst = xr.Dataset(
            {
                "dflx_sp_711": (("time", "wvl"), values_711),
                "dflx_sp_713": (("time", "wvl"), values_713),
                "dflx_sp_wiser": (("time", "wvl"), values_merge),
                "sensor_exposure_711": ("time", exposure_711),
                "sensor_exposure_713": ("time", exposure_713),
                "sensor_temperature_711": ("time", temp_711),
                "sensor_temperature_713": ("time", temp_713),
                "sensor_power_711": ("time", pwr_711),
                "sensor_power_713": ("time", pwr_713)
            },
            coords={
                "time": ("time", time),
                "wvl": ("wvl", wvls),
            }
        )

        if new:
            ds = dst.copy()
            new = False
        else:
            ds = xr.concat((ds, dst), dim='time')

    if new:
        return None

    # add meta
    for var in ds.variables:
        if (var == "time") or (var == "wvl") or (var.endswith("wiser")):
            continue
        i = 0 if var.endswith("711") else 1

        meta = taro.utils.meta_lookup(config, troposID=config["wiser_ids"][i])
        # drop calibration meta
        meta = keyfilter(lambda x: not x.startswith('calibration'), meta)
        # drop None values
        meta = valfilter(lambda x: x is not None, meta)
        # update netcdf attributes
        ds[var].attrs.update(meta)

    # add coordinats if available in config
    if config["coordinates"] is not None:
        lat, lon, alt = config["coordinates"]
        if lat is not None:
            ds["lat"] = lat
        if lon is not None:
            ds["lon"] = lon
        if alt is not None:
            ds["altitude"] = alt

    # add global meta data
    now = pd.to_datetime(np.datetime64("now"))
    gattrs.update({
        'processing_level': 'l1a',
        'product_version': taro.__version__,
        'history': f'{now.isoformat()}: Generated level l1a  by taro version {taro.__version__}; ',
    })
    ds.attrs.update(gattrs)

    # drop occurrence of duplicate sample values
    ds = ds.drop_duplicates("time")

    # add global coverage attributes
    ds = taro.futils.update_coverage_meta(ds, timevar="time")

    # add attributes to Dataset
    for k, v in vattrs.items():
        # iterate over suffixed variables
        for ki in [key for key in ds if key.startswith(k)]:
            ds[ki].attrs.update(v)

    # add encoding to Dataset
    ds = taro.futils.add_encoding(ds, vencode)

    return ds