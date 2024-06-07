import sys
import os


def main(cfg):
    print(cfg)
   
    if cfg == "train":
        os.system("python summarization/scripts/train.py")


    elif cfg == "infer":
         os.system("python summarization/scripts/infer.py")

    elif cfg == "test":
        os.system("python bot.py")



if __name__ == "__main__":
    main(sys.argv[1])