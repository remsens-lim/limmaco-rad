import shutil
import os.path
import warnings

import click
import logging
import parse
import xarray as xr
import pandas as pd
import numpy as np
import datetime as dt
import importlib.resources
import matplotlib as mpl
import matplotlib.pyplot as plt
import windrose

import taro
import taro.utils
import taro.futils
import taro.data
import taro.plot
import taro.keogram


logger = logging.getLogger(__name__)

DEFAULT_CONFIG = fn_config = os.path.join(
        importlib.resources.files("taro"),
        "conf/taro_config.json"
    )
def _configure(config):
    if config is None:
        config = taro.utils.get_default_config()
    else:
        config = taro.utils.read_json(os.path.abspath(config))
        config = taro.utils.merge_config(config)
    return config

# initialize commandline interface
@click.version_option(taro.__version__, package_name="limmaco-rad")
@click.group("taro")
def cli():
    pass

##########################
# Raw data handling
##########################
@cli.command("raw2daily")
@click.argument("input_path", nargs=1)
@click.argument("output_path", nargs=1)
@click.option("--config", "-c", type=click.Path(dir_okay=False, exists=True),
              help="Config file - will merge and override the default config.")
def raw2daily(input_path: str,
              output_path: str,
              config):
    """
    Moves data lines from the  continuous Loggernet files within input path (table names specified in config) to daily files in output path.
    The output path is suffixed according to config['path_sfx'] (default is '{dt:%Y/%m/}').
    The daily output files are named according to config['fname_out'] (default is '{dt:%Y-%m-%d}_taro-core_{campaign}_{table}_{resolution}_{datalvl}.c{collection:02d}.{sfx}').
    """
    config = _configure(config)
    taro.utils.init_logger(config)

    logger.info("Call taro.futils.raw2daily")
    taro.futils.raw2daily(
       inpath=os.path.abspath(input_path),
       outpath=os.path.abspath(output_path),
       config=config
    )

##########################
# Processing functions
##########################
@cli.group("process")
def process():
    print("Process")

@process.command("l1a")
@click.argument("input_files", nargs=-1)
@click.argument("output_path", nargs=1)
@click.option("--skip-exists", is_flag=True, help="Skip if output exists.")
@click.option("--config", "-c", type=click.Path(dir_okay=False, exists=True),
              help="Config file - will merge and override the default config.")
def process_l1a(input_files,
        output_path: str,
        skip_exists: bool,
        config: str):
    """
    Process input files (LoggerNet TOA5 ASCII Table Data) to rsd-car netcdf.
    """
    config = _configure(config)
    taro.utils.init_logger(config)

    with click.progressbar(input_files, label='Processing to l1a:') as files:
        for fn in files:
            fname_info = parse.parse(
                config["fname_out"],
                os.path.basename(fn)
            ).named

            fname_info.update({
                "datalvl": "l1a",
                "sfx": "nc"
            })

            outfile = os.path.join(output_path,
                                   "{dt:%Y/%m/}",
                                   config['fname_out'])
            outfile = outfile.format_map(fname_info)
            if skip_exists and os.path.exists(outfile):
                continue

            logger.info("Call taro.data.to_l1a")
            ds = taro.data.to_l1a(
               fname=fn,
               config=config
            )
            if ds is None:
                logger.warning(f"Skip {fn}.")
                continue

            taro.futils.to_netcdf(
                ds=ds,
                fname=outfile,
                timevar="time"
            )

@process.command("l1b")
@click.argument("input_files", nargs=-1)
@click.argument("output_path", nargs=1)
@click.option("--skip-exists", is_flag=True, help="Skip if output exists.")
@click.option("--resolution","-r",
              default='1min', show_default=True,
              help="Time resolution of output file.")
@click.option("--config", "-c", type=click.Path(dir_okay=False, exists=True),
              help="Config file - will merge and override the default config.")
def process_l1b(input_files,
                output_path: str,
                skip_exists: bool,
                resolution:str,
                config: str):
    """
    Process input files (taro l1a) to taro l1b files.
    Input files require at least the Radiation table (SensorStatus and Meteorologie are then assumed to be in the same directory).
    Flux variables are calibrated and corrected for sensor temperature.
    Dataset is resampled to desired resolution.
    Sun coordinate parameters are added.
    """
    config = _configure(config)
    taro.utils.init_logger(config)

    fdates = []
    for fn in input_files:
        finfo = parse.parse(
            config["fname_out"],
            os.path.basename(fn)
        )
        fdates.append(finfo["dt"])
    udays, indices = np.unique(fdates, return_inverse=True)


    with click.progressbar(udays, label='Processing to l1b:') as days:
        for i, day in enumerate(days):
            ifiles = np.argwhere(indices == i).ravel()
            files = [input_files[j] for j in ifiles]

            fname_info = parse.parse(
                config["fname_out"],
                os.path.basename(files[0])
            ).named
            fname_info.update({
                "table": "complete",
                "resolution": resolution,
                "datalvl": "l1b",
            })
            outfile = os.path.join(output_path,
                                   "{dt:%Y/%m/}",
                                   config['fname_out'])
            outfile = outfile.format_map(fname_info)
            if skip_exists and os.path.exists(outfile):
                continue

            ds_l1a = taro.futils.merge_with_rename(
                [xr.load_dataset(fn) for fn in files],
                dim="time",
                override=['lat','lon','altitude']
            )
            ds_l1a = ds_l1a.interpolate_na('time')

            logger.info("Call taro.data.to_l1b")
            ds = taro.data.to_l1b(
                ds_l1a,
                resolution=resolution,
                config=config
            )
            if ds is None:
                logger.warning(f"Skip {fn}.")
                continue

            taro.futils.to_netcdf(
                ds=ds,
                fname=outfile,
                timevar="time"
            )

###########################
# taro plot
###########################
@cli.group("quicklook")
def quickook():
    print("Make quicklooks")

@quickook.command("data")
@click.argument("input_files", nargs=-1)
@click.argument("output_path", nargs=1)
@click.option("--skip-exists", is_flag=True, help="Skip if output exists.")
@click.option("--config", "-c", type=click.Path(dir_okay=False, exists=True),
              help="Config file - will merge and override the default config.")
@click.option("--dpi", type=int, nargs=1,
              default=300, show_default=True,
              help="DPI for output png-file.")
def ql_data(input_files, output_path, skip_exists, config,dpi):
    config = _configure(config)
    taro.utils.init_logger(config)

    with click.progressbar(input_files, label='Make daily data quicklooks:') as files:
        for fn in files:
            fname_info = parse.parse(
                config["fname_out"],
                os.path.basename(fn)
            ).named

            fname_info.update({
                "table": "data",
                "sfx": "png"
            })

            outfile = os.path.join(output_path,
                                   "{dt:%Y/%m/}",
                                   config['fname_out'])
            outfile = outfile.format_map(fname_info)
            if skip_exists and os.path.exists(outfile):
                continue

            ds_l1b = xr.load_dataset(fn)
            fig = plt.figure(figsize=(8, 12))
            axs = [
                fig.add_subplot(311),
                fig.add_subplot(312),
                fig.add_subplot(313,axes_class=windrose.WindAxes)
            ]

            plots = ds_l1b.quicklooks.flux(ax=axs[0])
            axs[0].set_ylim([-10, 1310])
            pl, (ax, pax, rax) = ds_l1b.quicklooks.meteorology(
                ax=axs[1],
                legend=False,
                ylim={"tair": [-10, 40],
                      "pair": [990, 1040],
                      "rh": None}
            )
            rax.legend(handles=pl, loc='lower right')
            ds_l1b.quicklooks.wind_barbs(ax=ax,y=35)

            ds_l1b.quicklooks.windrose(ax=axs[2])

            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            fig.savefig(outfile, dpi=dpi, bbox_inches='tight')
            plt.close(fig)

@quickook.command("quality")
@click.argument("input_files", nargs=-1)
@click.argument("output_path", nargs=1)
@click.option("--skip-exists", is_flag=True, help="Skip if output exists.")
@click.option("--config", "-c", type=click.Path(dir_okay=False, exists=True),
              help="Config file - will merge and override the default config.")
@click.option("--dpi", type=int, nargs=1,
              default=300, show_default=True,
              help="DPI for output png-file.")
def ql_quality(input_files: list, output_path: str, skip_exists:bool, config: dict, dpi: int):
    config = _configure(config)
    taro.utils.init_logger(config)

    with click.progressbar(input_files, label='Make daily quality quicklooks:') as files:
        for fn in files:
            fname_info = parse.parse(
                config["fname_out"],
                os.path.basename(fn)
            ).named

            fname_info.update({
                "table": "quality",
                "sfx": "png"
            })

            outfile = os.path.join(output_path,
                                   "{dt:%Y/%m/}",
                                   config['fname_out'])
            outfile = outfile.format_map(fname_info)
            if skip_exists and os.path.exists(outfile):
                continue

            ds_l1b = xr.load_dataset(fn)
            fig, axs = plt.subplots(4, 1, figsize=(10, 12), constrained_layout=True,
                                    gridspec_kw={"height_ratios": [2, 1, 1, 2]})

            pl_status, _ = ds_l1b.quicklooks.status(ax=axs[0])
            pl_align, _ = ds_l1b.quicklooks.sensor_alignment(ax=axs[1])
            #ds_l1b.quicklooks.quality_range_dhi2ghi(ax=axs[1], ratio=True, kwargs={'alpha': 0.5})
            #ds_l1b.quicklooks.quality_range_shading(ax=axs[1], ratio=True, kwargs={'alpha': 0.5, 'hatch': '//'})
            #ds_l1b.quicklooks.quality_range_lwd2temp(ax=axs[1], ratio=True, kwargs={'alpha': 0.5})

            #axs[1].legend(bbox_to_anchor=(1, 1))
            #axs[1].set_ylim([0.2, 1.6])

            ds_l1b.quicklooks.solar_at_night(ax=axs[2])
            axs[2].legend(bbox_to_anchor=(1, 1))

            pl_flags = ds_l1b.quicklooks.status_flags(ax=axs[3], freq='15min')

            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            fig.savefig(outfile, dpi=dpi, bbox_inches='tight')
            plt.close(fig)

###########################
# taro info
###########################
@cli.command("info")
@click.argument("ids", nargs=-1)
@click.option("--serial",is_flag=True, help="Print only serial number.")
@click.option("--tropos",is_flag=True, help="Print only TROPOS ID.")
@click.option("--device",is_flag=True, help="Print only device information.")
@click.option("--calibration", is_flag=True, help="Print only calibration information. If 'ids' are not set and --calibration is set: Prints calibration due overview of all instruments.")
@click.option("--config", "-c", type=click.Path(dir_okay=False, exists=True),
              help="Config file - will merge and override the default config.")
def info(ids:str, calibration:bool, serial:bool, tropos:bool, device:bool, config: dict):
    """Print taro device information. If no option is selected, full information is printed.
    """
    config = _configure(config)
    if len(ids) == 0:
        cdate = dt.datetime.now()
        mapping = taro.utils.read_json(config["file_instrument_map"])
        ids, stations, ctimes = [],[],[]
        for id in mapping:
            res = taro.utils.parse_calibration(
                config["file_calibration"],
                troposID=id,
                cdate=cdate,
            )
            if res is None:
                if calibration:
                    # skip if no calibration information
                    continue
                ctimes.append(np.datetime64("NaT"))
            else:
                ctimes.append(res.time.values)
            ids.append(id)
            stations.append(mapping[id]['station'])
        times = np.array(ctimes).astype("datetime64[M]") + np.timedelta64(24, 'M')

        if calibration:
            click.echo("Calibrations due:")
            isort = np.argsort(ctimes)
            for i in isort:
                timedelta = pd.to_datetime(times[i]) - cdate#.astype("datetime64[M]")
                text = f"{pd.to_datetime(times[i]):%Y-%m}: {ids[i]} - {stations[i]} - {mapping[ids[i]]['device']}"
                if timedelta < dt.timedelta(days=0):
                    click.echo(click.style(text, fg='red'))
                elif timedelta < dt.timedelta(days=30*6):#np.timedelta64(6, 'M'):
                    click.echo(click.style(text, fg='yellow'))
                else:
                    click.echo(text)
        else:
            isort = np.argsort(stations)
            tmpstation = ""
            for i in isort:
                if stations[i]!=tmpstation:
                    click.echo(stations[i])
                    tmpstation = stations[i]
                text = f"    {ids[i]}: {mapping[ids[i]]['device']}"
                if not np.isnat(times[i]):
                    texttime =f" -  calibration due: {pd.to_datetime(times[i]):%Y-%m}"
                    if times[i] < np.datetime64("now"):
                        texttime = click.style(texttime,fg='red')
                    text += texttime
                click.echo(text)
        return

    # if nothing is set, echo all
    if not (serial or tropos or device or calibration):
        serial, tropos, device, calibration = True, True, True, True

    for i, id in enumerate(ids):
        if i != 0:
            click.echo("")

        if id.startswith('A2'):
            meta = taro.utils.meta_lookup(config, troposID=id)
        else:
            meta = taro.utils.meta_lookup(config, serial=id)

        if tropos:
            click.echo(f"TROPOS ID: {meta['troposID']}")
        if serial:
            click.echo(f"Serial No: {meta['serial']}")
        if device:
            click.echo(f"Device:    {meta['station']} - {meta['device']}")
        if calibration and meta['calibration_date'] is not None:
            if "temperature_correction_coef" in meta:
                click.echo("Temperature correction coefficients:")
                click.echo(f"    {meta['temperature_correction_coef']}")
            click.echo("Calibrations:")
            for d, f, e in zip(meta['calibration_date'],
                               meta['calibration_factor'],
                               meta['calibration_error']):
                click.echo(f"    {d}: {f:6.3f} ({meta['calibration_factor_units']}) +- {e:4.2f} ({meta['calibration_error_units']})")
            nextd = pd.to_datetime(meta['calibration_date'][-1]) + dt.timedelta(days=366*2)
            click.echo(f"Calibration due: {nextd:%Y-%m}!")



@click.version_option(taro.__version__, package_name="limmaco-rad")
@click.group("asi16")
def cli_asi16():
    pass


# def asi16_missing_dates(
#         images_path: str,
#         processed_path: str,
#         config: dict,
#         raw: bool = True
# ):
#     fname_tmp = config['asi16_raw'] if raw else config['asi16_out']
#     img_files = []
#     img_dates = []
#     for p, d, f in os.walk(images_path):
#         img_files += [os.path.join(p, fi) for fi in f if fi.endswith(".jpg")]
#     for fn in img_files:
#         finfo = parse.parse(fname_tmp, os.path.basename(fn)).named
#         img_dates.append(finfo["dt"])
#     img_dates = np.unique(img_dates)
#
#     pro_files = []
#     pro_dates = []
#     for p, d, f in os.walk(processed_path):
#         pro_files += [os.path.join(p, fi) for fi in f if fi.endswith(".bmp")]
#     for fn in pro_files:
#         finfo = parse.parse(fname_tmp, os.path.basename(fn)).named
#         pro_dates.append(finfo["dt"])
#     pro_dates = np.unique(pro_dates)
#
#     missing_dates = np.setdiff1d(img_dates, pro_dates, assume_unique=True)
#     return missing_dates


@cli_asi16.command("move")
@click.argument("image_files", nargs=-1)
@click.argument("out_path", nargs=1)
@click.option("--raw/--no-raw",
              default=True, show_default=True,
              help="Defines the type of image_path data.")
@click.option("--config", "-c", type=click.Path(dir_okay=False, exists=True),
              help="Config file - will merge and override the default config.")
def asi16_move(
        image_files: list,
        out_path: str,
        raw: bool,
        config: str
):
    config = _configure(config)
    fname_tmp = config['asi16_raw'] if raw else config['asi16_out']

    nw=0
    for fn in image_files:
        finfo = parse.parse(fname_tmp, os.path.basename(fn))
        if finfo is None:
            if nw < 2:
                warnings.warn(f"{os.path.basename(fn)} can't be parsed against {fname_tmp}.")
                nw += 1
            if nw == 2:
                warnings.warn(f"More warnings suppressed -  Can't parse against {fname_tmp}.")
            continue
        finfo = finfo.named
        if not raw:
            # move unprocessed to raw file structure
            path_out = os.path.join(out_path, f"{finfo['dt']:%Y%m%d/}")
        else:
            # move unprocessed to a l0 file structure
            path_out = os.path.join(out_path, f"{finfo['dt']:%Y/%m/%d/}")

        fname_out_tmp = config['asi16_raw'] if not raw else config['asi16_out']
        fname_out = fname_out_tmp.format(
            dt=finfo["dt"],
            shot=finfo["shot"],
            sfx=finfo["sfx"],
            **config
        )
        os.makedirs(path_out, exist_ok=True)
        shutil.move(fn, os.path.join(path_out, fname_out))

def asi16_missing_dates(
        images_path: str,
        processed_path: str,
        config: dict,
        raw: bool = True
):
    fname_tmp = config['asi16_raw'] if raw else config['asi16_out']
    img_files = []
    for p, d, f in os.walk(images_path):
        img_files += [os.path.join(p, fi) for fi in f if fi.endswith(".jpg")]

    pro_files = []
    pro_basenames = []
    for p, d, f in os.walk(processed_path):
        pro_files += [os.path.join(p, fi) for fi in f if fi.endswith(".bmp")]
        pro_basenames += [fi for fi in f if fi.endswith(".bmp")]

    missing_processing = []
    missing_dates = []
    processed_images = []
    processed_results = []
    nw=0
    for fn in img_files:
        finfo = parse.parse(fname_tmp, os.path.basename(fn))
        if finfo is None:
            if nw < 2:
                warnings.warn(f"{os.path.basename(fn)} can't be parsed against {fname_tmp}.")
                nw += 1
            if nw == 2:
                warnings.warn(f"More warnings suppressed -  Can't parse against {fname_tmp}.")
            continue
        finfo = finfo.named
        finfo.update({"sfx":"bmp"})
        if fname_tmp.format(**finfo) not in pro_basenames:
            missing_processing.append(fn)
            missing_dates.append(finfo["dt"])
        else:
            idx = pro_basenames.index(fname_tmp.format(**finfo))
            processed_images.append(fn)
            processed_results.append(pro_files[idx])

    return np.array(missing_dates), missing_processing, processed_images, processed_results

@cli_asi16.command("status")
@click.argument("images_path", nargs=1)
@click.argument("processed_path", nargs=1)
@click.option("--config", "-c", type=click.Path(dir_okay=False, exists=True),
              help="Config file - will merge and override the default config.")
@click.option("--raw/--no-raw",
              default=True, show_default=True,
              help="Defines the type of data to be checked on processing status.")
def asi16_check_processed(
        images_path: str,
        processed_path: str,
        config: str,
        raw: bool
):

    config = _configure(config)
    missing_dates, _, _, _ = asi16_missing_dates(
        images_path=images_path,
        processed_path=processed_path,
        config=config,
        raw=raw
    )
    mdates, counts = np.unique(missing_dates.astype("datetime64[D]"), return_counts=True)

    print("Unprocessed cases:")
    print("date      , cases")
    for date, count in zip(mdates,counts):
        print(f"{date}, {count:4d}")

@cli_asi16.command("move-unprocessed")
@click.argument("images_path", nargs=1)
@click.argument("processed_path", nargs=1)
@click.argument("out_path", nargs=1)
@click.option("--raw/--no-raw",
              default=True, show_default=True,
              help="Defines the type of data to be checked on processing status.")
@click.option("--config", "-c", type=click.Path(dir_okay=False, exists=True),
              help="Config file - will merge and override the default config.")
def asi16_move_unprocessed(
        images_path: str,
        processed_path: str,
        out_path: str,
        raw: bool,
        config: str
):
    config = _configure(config)
    fname_tmp = config['asi16_raw'] if raw else config['asi16_out']
    missing_dates, missing_processing, _, _ = asi16_missing_dates(
        images_path=images_path,
        processed_path=processed_path,
        config=config,
        raw=raw
    )

    for fn in missing_processing:
        finfo = parse.parse(fname_tmp, os.path.basename(fn)).named
        if not raw:
            # move unprocessed to raw file structure
            path_out = os.path.join(out_path, f"{finfo['dt']:%Y%m%d/}")
        else:
            # move unprocessed to a l0 file structure
            path_out = os.path.join(out_path, f"{finfo['dt']:%Y/%m/%d/}")

        fname_out_tmp = config['asi16_raw'] if not raw else config['asi16_out']
        fname_out = fname_out_tmp.format(
            dt=finfo["dt"],
            shot=finfo["shot"],
            sfx=finfo["sfx"],
            **config
        )
        os.makedirs(path_out, exist_ok=True)
        shutil.move(fn, os.path.join(path_out, fname_out))

@cli_asi16.command("move-processed")
@click.argument("images_path", nargs=1)
@click.argument("processed_path", nargs=1)
@click.argument("out_path_img", nargs=1)
@click.argument("out_path_pro", nargs=1)
@click.option("--raw/--no-raw",
              default=True, show_default=True,
              help="Defines the type of data to be checked on processing status.")
@click.option("--config", "-c", type=click.Path(dir_okay=False, exists=True),
              help="Config file - will merge and override the default config.")
def asi16_move_processed(
        images_path: str,
        processed_path: str,
        out_path_img: str,
        out_path_pro:str,
        raw: bool,
        config: str
):
    config = _configure(config)
    fname_tmp = config['asi16_raw'] if raw else config['asi16_out']
    missing_dates, _, img_files, processed_files = asi16_missing_dates(
        images_path=images_path,
        processed_path=processed_path,
        config=config,
        raw=raw
    )

    for fn,fnp in zip(img_files, processed_files):
        finfo = parse.parse(fname_tmp, os.path.basename(fn)).named
        finfo_pro = parse.parse(fname_tmp, os.path.basename(fnp)).named
        if not raw:
            # move unprocessed to raw file structure
            path_out = os.path.join(out_path_img, f"{finfo['dt']:%Y%m%d/}")
            path_out_pro = os.path.join(out_path_pro, f"{finfo_pro['dt']:%Y%m%d/}")
        else:
            # move unprocessed to a l0 file structure
            path_out = os.path.join(out_path_img, f"{finfo['dt']:%Y/%m/%d/}")
            path_out_pro = os.path.join(out_path_pro, f"{finfo_pro['dt']:%Y/%m/%d/}")

        fname_out_tmp = config['asi16_raw'] if not raw else config['asi16_out']
        fname_out = fname_out_tmp.format(
            dt=finfo["dt"],
            shot=finfo["shot"],
            sfx=finfo["sfx"],
            **config
        )
        fname_out_pro = fname_out_tmp.format(
            dt=finfo_pro["dt"],
            shot=finfo_pro["shot"],
            sfx=finfo_pro["sfx"],
            **config
        )
        os.makedirs(path_out, exist_ok=True)
        os.makedirs(path_out_pro, exist_ok=True)
        shutil.move(fn, os.path.join(path_out, fname_out))
        shutil.move(fnp, os.path.join(path_out_pro, fname_out_pro))


@cli_asi16.command("keogram")
@click.argument("images", nargs=-1)
@click.argument("keogram_filename", nargs=1)
@click.option("--cffile", type=str, default=None, show_default=True,
              help="Filename of cloudiness file.")
@click.option("--lon",type=float, default=None, show_default=True,
              help="Longitude coordinate (degrees East) of the image. If None, try to parse longitude from config.")
@click.option("--lat",type=float, default=None, show_default=True,
              help="Latitude coordinate (degrees North) of the image. If None, try to parse latitude from config.")
@click.option("-r","--radius-scale",type=float,default=1.,show_default=True,
              help="Radius ratio to crop the picture.")
@click.option("-a","--angle-offset", type=float, default=0, show_default=True,
              help="Static angle to rotate the picture counter-clockwise.")
@click.option("--flip/--no-flip",help="Flip the image before rotating.")
@click.option("--fill-color",nargs=3,default=[0,0,0],show_default=True,
              help="Color of missing images in keogram (R,G,B).")
@click.option("--whole-day/--no-whole-day",default=False)
@click.option("--config", "-c", type=click.Path(dir_okay=False, exists=True),
              help="Config file - will merge and override the default config.")
def asi16_keogram(
        images: list,
        keogram_filename: str,
        cffile: str,
        lon: float,
        lat: float,
        radius_scale: float,
        angle_offset: float,
        flip: bool,
        fill_color: list,
        whole_day:bool,
        config: str
):
    config = _configure(config)

    if lon is None:
        if config["coordinates"] is None:
            warnings.warn("No coordinates in config - proceed with longitude=None")
            longitude = None
        else:
            longitude = config["coordinates"][1]
    else:
        longitude = lon

    if lat is None:
        if config["coordinates"] is None:
            warnings.warn("No coordinates in config - proceed with latitude=None")
            latitude = None
        else:
            latitude = config["coordinates"][0]
    else:
        latitude = lat

    img_dates = []
    for fn in images:
        finfo = parse.parse(config['asi16_out'], os.path.basename(fn)).named
        img_dates.append(finfo["dt"])

    keogram = taro.keogram.make_keogram(
        img_files=images,
        img_dates=img_dates,
        longitude=longitude,
        latitude=latitude,
        radius_scale=radius_scale,
        angle_offset=angle_offset,
        flip=flip,
        fill_color=fill_color,
        whole_day=whole_day
    )

    if cffile is not None:
        dscf = xr.load_dataset(cffile)
        cf = dscf.cloudiness.mean(dim="exposure_key", skipna=True).squeeze()

    mpl.use('Agg')
    if cffile is not None:
        fig = plt.figure(figsize=(10, 7))
        gs = fig.add_gridspec(nrows=7, ncols=1, wspace=0, hspace=0.05)
        ax_keo = fig.add_subplot(gs[1:, 0])
        ax_cf = fig.add_subplot(gs[0, 0], sharex=ax_keo)
        ax_cf = taro.plot.cloudfraction(cf, Nsmooth=10, ax=ax_cf)
    else:
        fig, ax_keo = plt.subplots(1,1, figsize=(10, 6))

    fig, ax_keo = taro.keogram.plot_keogram(
        keogram,
        sdate=img_dates[0],
        edate=img_dates[-1],
        ax=ax_keo
    )
    fig.savefig(keogram_filename, dpi=300, bbox_inches='tight')

@cli_asi16.command("test-config")
@click.argument("image", nargs=1)
@click.option("--lon",type=float, default=None, show_default=True,
              help="Longitude coordinate (degrees East) of the image. If None, try to parse longitude from config.")
@click.option("--lat",type=float, default=None, show_default=True,
              help="Latitude coordinate (degrees North) of the image. If None, try to parse latitude from config.")
@click.option("-r","--radius-scale",type=float,default=1.,show_default=True,
              help="Radius ratio to crop the picture.")
@click.option("-a","--angle-offset", type=float, default=0, show_default=True,
              help="Static angle to rotate the picture counter-clockwise.")
@click.option("--flip/--no-flip",help="Flip the image before rotating.")
@click.option("--config", "-c", type=click.Path(dir_okay=False, exists=True),
              help="Config file - will merge and override the default config.")
def asi16_test_config(
        image: str,
        lon: float,
        lat: float,
        radius_scale: float,
        angle_offset: float,
        flip: bool,
        config: str
):
    config = _configure(config)

    if lon is None:
        if config["coordinates"] is None:
            warnings.warn("No coordinates in config - proceed with longitude=None")
            longitude = None
        else:
            longitude = config["coordinates"][1]
    else:
        longitude = lon

    if lat is None:
        if config["coordinates"] is None:
            warnings.warn("No coordinates in config - proceed with latitude=None")
            latitude = None
        else:
            latitude = config["coordinates"][0]
    else:
        latitude = lat

    finfo = parse.parse(config['asi16_out'], os.path.basename(image)).named
    image_out = taro.keogram.test_image_config(
        img_file=image,
        img_date=finfo["dt"],
        longitude=longitude,
        latitude=latitude,
        radius_scale=radius_scale,
        angle_offset=angle_offset,
        flip=flip,
    )
    image_out.show()


@click.version_option(taro.__version__, package_name="limmaco-rad")
@click.group("wiser")
def cli_wiser():
    pass

@cli_wiser.command("l1a")
@click.argument("input_files", nargs=-1)
@click.argument("output_path", nargs=1)
@click.option("--skip-exists", is_flag=True, help="Skip if output exists.")
@click.option("--config", "-c", type=click.Path(dir_okay=False, exists=True),
              help="Config file - will merge and override the default config.")
def wiser_l1a(input_files,
        output_path: str,
        skip_exists: bool,
        config: str):
    """
    Process input files to rsd-car netcdf.
    """
    config = _configure(config)
    taro.utils.init_logger(config)

    dates = []
    pfs = []
    for fn in input_files:
        fname_info = parse.parse(
            config["wiser_raw"],
            os.path.basename(fn)
        ).named
        pfs.append(os.path.dirname(os.path.abspath(fn)))
        dates.append(np.datetime64(fname_info["dt"]))
    dates, idx = np.unique(np.array(dates).astype("datetime64[D]"), return_index=True)
    pfs = np.array(pfs)[idx]
    print(dates)
    with click.progressbar(np.arange(len(dates)), label='Processing wiser l0 to l1a:') as idxs:
        for i in idxs:
            date = pd.to_datetime(dates[i])
            pf = pfs[i]

            fname_info = {
                "dt": date, "campaign": config["campaign"], "sfx": "nc"
            }
            outfile = os.path.join(output_path,
                                   "{dt:%Y/%m/}",
                                   config['wiser_out'])
            outfile = outfile.format_map(fname_info)
            if skip_exists and os.path.exists(outfile):
                continue

            logger.info("Call taro.data.wiser_to_l1a")
            ds = taro.data.wiser_to_l1a(date=date, pf=pf, config=config)

            if ds is None:
                logger.warning(f"Skip {fn}.")
                continue

            taro.futils.to_netcdf(
                ds=ds,
                fname=outfile,
                timevar="time"
            )

@cli_wiser.command("quicklook")
@click.argument("input_files", nargs=-1)
@click.argument("output_path", nargs=1)
@click.option("--skip-exists", is_flag=True, help="Skip if output exists.")
@click.option("--config", "-c", type=click.Path(dir_okay=False, exists=True),
              help="Config file - will merge and override the default config.")
@click.option("--dpi", type=int, nargs=1,
              default=300, show_default=True,
              help="DPI for output png-file.")
def wiser_quicklook(input_files,
        output_path: str,
        skip_exists: bool,
        config: str,
        dpi:int):
    """
    Make wiser quicklooks.
    """
    config = _configure(config)
    taro.utils.init_logger(config)

    with click.progressbar(input_files, label='Make daily data quicklooks:') as files:
        for fn in files:
            fname_info = parse.parse(
                config["wiser_out"],
                os.path.basename(fn)
            ).named

            fname_info.update({
                "sfx": "png"
            })

            outfile = os.path.join(output_path,
                                   "{dt:%Y/%m/}",
                                   config['wiser_out'])
            outfile = outfile.format_map(fname_info)
            if skip_exists and os.path.exists(outfile):
                continue

            ds = xr.load_dataset(fn)
            fig,axs = taro.plot.wiser_quicklook(ds)

            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            fig.savefig(outfile, dpi=dpi, bbox_inches='tight')
            plt.close(fig)