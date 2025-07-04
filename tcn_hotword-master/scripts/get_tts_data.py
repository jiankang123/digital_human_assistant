import sys
import os
import requests
from tqdm import tqdm
import multiprocessing as mlp

EN_SPKS = [
    "mars_william",
    "mars_wendy",
    "mars_olivia",
    "mars_luna",
    "mars_emily",
    "mars_luca",
    "mars_harry",
    "mars_eric",
    "mars_andy",
    "mars_abby",
    "mercury_aria",
    "mercury_jenny",
    "mercury_guy",
    "mercury_aria@narration-professional",
    "mercury_aria@empathetic",
    "mercury_aria@cheerful",
    "mercury_aria@customerservice",
    "mercury_aria@chat",
    "mercury_aria@newscast-formal",
    "mercury_aria@newscast-casual",
    "mercury_jenny@assistant",
    "mercury_jenny@chat",
    "mercury_jenny@customerservice",
    "mercury_jenny@newscast",
    "mercury_guy@newscast",
    "mercury_clara",
    "mercury_liam",
    "mercury_libby",
    "mercury_ryan",
    "mercury_mia",
    "mercury_sam",
    "mercury_yan",
    "mercury_connor",
    "mercury_emily",
    "mercury_neerja",
    "mercury_prabhat",
    "mercury_mitchell",
    "mercury_molly",
    "mercury_james",
    "mercury_luna",
    "mercury_wayne",
    "mercury_amber",
    "mercury_ana",
    "mercury_brandon",
    "mercury_christopher",
    "mercury_cora",
    "mercury_elizabeth",
    "mercury_eric",
    "mercury_jacob",
    "mercury_michelle",
    "mercury_monica",
    "mercury_leah",
    "mercury_ashley",
    "mercury_sonia",
    "mercury_asilia",
    "mercury_chilemba",
    "mercury_elimu",
    "mercury_imani",
    "mercury_abeo",
    "mercury_ezinne",
    "mercury_luke",
    "mercury_jenny@angry",
    "mercury_jenny@cheerful",
    "mercury_jenny@sad",
    "mercury_jenny@excited",
    "mercury_jenny@friendly",
    "mercury_jenny@terrified",
    "mercury_jenny@shouting",
    "mercury_jenny@unfriendly",
    "mercury_jenny@whispering",
    "mercury_jenny@hopeful",
    "mercury_guy@cheerful",
    "mercury_guy@angry",
    "mercury_guy@sad",
    "mercury_guy@excited",
    "mercury_guy@friendly",
    "mercury_guy@terrified",
    "mercury_guy@shouting",
    "mercury_guy@unfriendly",
    "mercury_guy@whispering",
    "mercury_guy@hopeful",
    "mercury_aria@excited",
    "mercury_aria@friendly",
    "mercury_aria@terrified",
    "mercury_aria@unfriendly",
    "mercury_aria@whispering",
    "mercury_aria@sad",
    "mercury_aria@angry",
    "mercury_aria@shouting",
    "mercury_sara@excited",
    "mercury_sara@friendly",
    "mercury_sara@hopeful",
    "mercury_sara@shouting",
    "mercury_sara@terrified",
    "mercury_sara@unfriendly",
    "mercury_sara@whispering",
    "mercury_aria@hopeful"
]

EN_TEXTS = [
    "mark that", "mark", "that", 
    "shut down camera", "shut down", "shut", "down camera", "camera",
    "start recording", "start", "recording",
    "stop recording", "stop",
    "take a photo", "take a", "take", "a photo", "photo"
]

YUE_SPKS = [
    "dora_lpcnet_24k",
    "mercury_hiugaai",
    "mars_shanshan",
    "mercury_hiumaan",
    "mercury_wanlung",
    "mars_jiajia",
]

YUE_TEXTS = [
    "粤语测试",
    "机器越来越小、便携、移动，人机交互方式必然会更智能、更高效",
    "定义下一代人机交互，让人和机器的交互更高效智能",
    "科技使人自由和社会进步",
    "我们要成为强执行力的创新型科技组织"
]

CHUAN_SPKS =[
    "baoer_lpcnet_24k",
    "mars_chuangirl",
    "moxiaoxi_lpcnet_24k",
    "mercury_sc-yunxi",
]

CHUAN_TEXTS = [
    "四川话测试",
    "机器越来越小、便携、移动，人机交互方式必然会更智能、更高效",
    "定义下一代人机交互，让人和机器的交互更高效智能",
    "科技使人自由和社会进步",
    "我们要成为强执行力的创新型科技组织"
]


 # text_list = ["做个标记", "关闭相机", "开始录像", "停止录像", "拍张照片"]
# text_list = ["做个标记", "做个", "做个标", "关闭相机", "关闭", "关闭相", 
#              "开始录像", "开始", "开始录", "停止录像", "停止", "停止录",
#              "拍张照片", "拍张", "拍张照"]

def get_wav(url, wav_path):
    response = requests.get(url)
    with open(wav_path, 'wb') as fout:
        fout.write(response.content)


def main():
    out_dir = sys.argv[1]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    trans = open(os.path.join(out_dir, "trans"), 'w', encoding='utf-8')

    print("preparing tasks ......")
    tasks = []
    pool = mlp.Pool(12)
    base_url = "http://platform-tts-test.mobvoi.com/api/synthesis?audio_type=wav"
    for spr in CHUAN_SPKS:
        for i, text in enumerate(CHUAN_TEXTS):
            url = base_url + "&speaker={}&text={}".format(spr, text)
            wav_name = spr.replace("@", "_") + '_' + str(i+1) + ".wav"
            wav_path = os.path.join(out_dir, wav_name)
            if not os.path.exists(wav_path):
                tasks.append(pool.apply_async(get_wav, (url, wav_path)))
                trans.write(wav_path + '\t' + text + '\n')
    trans.close()
    print("doing tasks ......")
    for i in tqdm(range(len(tasks))):
        tasks[i].get()
    print("done")

if __name__ == "__main__":
    main()