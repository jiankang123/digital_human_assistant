import sys
import os
import subprocess

def apply_oners_kws(wav_in, wav_kws):
    exe_path = "/ssd1/kai.zhou/workspace/dsp/sdk/old_build.embedded_linux-x86_64-oners.Release"
    cmd = "{}/mobvoi_dsp_insta360_demo 3 1 0 16 0 0 0 {} 1 {}".format(exe_path, wav_in, wav_kws)
    subprocess.check_output(cmd, shell=True)


def apply_onex2_kws(wav_in, wav_kws):
    exe_path = "/ssd1/kai.zhou/workspace/dsp/sdk/old_build.embedded_linux-x86_64-onex2.Release"
    cmd = "{}/mobvoi_dsp_insta360_demo 1 1 0 16 0 0 0 {} 1 {}".format(exe_path, wav_in, wav_kws)
    subprocess.check_output(cmd, shell=True)


def apply_onex3_kws(wav_in, wav_kws):
    exe_path = "/ssd1/kai.zhou/workspace/dsp/sdk/old_build.embedded_linux-x86_64-onex3.Release"
    cmd = "{}/mobvoi_dsp_insta360_demo 4 1 0 16 0 0 0 {} 1 {}".format(exe_path, wav_in, wav_kws)
    subprocess.check_output(cmd, shell=True)