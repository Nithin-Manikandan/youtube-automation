import shutil

import imageio_ffmpeg


def exe():
    return shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
