import sys, os


def main():
    in_dir = sys.argv[1]
    for root, dirs, files in os.walk(in_dir):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                # 去除文件名中的空格
                out_path = file_path.replace(' ', '')
                os.rename(file_path, out_path)
    print("Done")

if __name__ == "__main__":
    main()
