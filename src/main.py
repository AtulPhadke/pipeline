import argparse
import threading
import sys
import time
import subprocess
import os
import keyboard
import tkinter
from tkinter import filedialog
from tkscrolledframe import ScrolledFrame
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons
from brukerapi.dataset import Dataset
from dipy.io.image import load_nifti, save_nifti
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel
from dipy.segment.mask import median_otsu
import dipy.reconst.dti as dti
from dipy.core.histeq import histeq
import nibabel as nib
import numpy as np
import time
import math
import SimpleITK as sitk

parser = argparse.ArgumentParser(description="Getting File Information")
parser.add_argument("-cmd", action="store_true")
args = parser.parse_args()

HOME = os.getcwd()

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'
SEPERATOR = "__________________________________"

CONVERSION = """
______________________________________

Convert bruker to nifti using 2DSEQ file
as well as additional parameter files.

‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
"""

SPLITING = """
______________________________________

Split nifti by an axis and convert 4D
to 3D Images.

‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
"""

DIFFUSION = """
_____________________________________________

Perform DTI Analysis using nifti image
and generate variable images (l1, adc, fa...)

‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
"""

SEGMENTATION = """
_____________________________________________

Segment a DTI image by using an atlas
combined with a scan registered with an ana-
tomy scan.

‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
"""

SAVE_FILE = """
_____________________________________________

Save files under new file name and into out-
put directory that was chosen.
_____________________________________________
"""

ARROW = """
 |
 v
"""

class Pipeline:
    def __init__(self):
        self.process =  {"Conversion": False, "Spliting": False, "DTI": False, "Segmentation": False}
        self.process_text = []
        self.current_dir = ""
        self.allFiles = []
        self.chosen_file = ""

        self.currentFileIndex = 0
        self.dirIndex = 0
        self.dirs = None
        self.fileOverflow = False
        self.f1p = ""
        self.nii = False
        self.new_name = ""
        self.OUTPUT_DIR = ""
        self.img = None
        self.LAMBDA = False
        self.FA = False
        self.ADC = False
        self.RD = False

    def __parse_file(self, directory):
        choosing_file = True
        while choosing_file:
            chosen_file = input()
            if chosen_file not in os.listdir(directory):
                print("Invalid File, try again: ", end="")
            else:
                return os.path.abspath(chosen_file)

    def __displayFirstFiles(self, allFiles):
        currentFileIndex = 0
        for file in allFiles:
            print("-- " + file)
            if currentFileIndex > 15:
                fileOverflow = True
                break
            currentFileIndex += 1

    def initiate_directory(self):
        self.currentFileIndex = 0
        self.current_dir = os.path.expanduser(self.current_dir)
        self.allFiles = []
        for file in os.listdir(self.current_dir):
            self.allFiles.append(file)
        print("Current Directory --> \033[34;1;4m"+ str(self.current_dir) + "\033[0m")
        self.fileOverflow = False
        for file in self.allFiles:
            if os.path.isdir(os.path.join(self.current_dir, file)):
                print(" \033[31;1;4m" + file + "\033[0m")
            else:
                print(" -- \033[1;32m" + file + "\033[0m")
            if self.currentFileIndex > 15:
                self.fileOverflow = True
                break
            self.currentFileIndex += 1
        if self.fileOverflow:
            print("## Press enter to see more files...", end='')

    def get_firstfilescmd(self):
        self.current_dir = os.path.expanduser("~")

        print("\n")

        print("To change directory type 'cd'.")
        print("To choose a file type 'chs'.\n")

        with Waiting_Spinner(delay=0.15, askingText="Press enter to continue: "):
            if input():
                pass

        print("Here's a list of all files:\n")
        choose = False
        self.chosen_file = False
        firstCycle = True

        numberOfDirs = 1


        self.initiate_directory()
        while not choose:
            if self.currentFileIndex != 0 and self.currentFileIndex < len(self.allFiles) - 1:
                userInput = input()

                if userInput == "":
                    sys.stdout.write(CURSOR_UP_ONE)
                    sys.stdout.write(ERASE_LINE)

                    if os.path.isdir(os.path.join(self.current_dir,self.allFiles[self.currentFileIndex+1])):
                        print(" \033[31;1;4m" + self.allFiles[self.currentFileIndex+1] + "\033[0m")
                    else:
                        print(" -- \033[1;32m" + self.allFiles[self.currentFileIndex+1] + "\033[0m")

                    print("## Press enter to see more files...", end='')
                    self.currentFileIndex += 1

                elif userInput == "chs":
                    print("Please pick your file now: ")
                    self.chosen_file = self.__parse_file(self.current_dir)
                    break
                elif userInput == "cd":
                    self.dirs = [ name for name in os.listdir(self.current_dir) if os.path.isdir(os.path.join(self.current_dir, name)) ]
                    print("\n________________________________________")
                    print("\nPress the 'W' key to cycle through directories. ")
                    print("Press the enter key to choose the new directory. \n")
                    self.__cycle_dirs()

                elif userInput == "goback":
                    print("\n"+SEPERATOR)
                    self.current_dir = os.path.join(self.current_dir, "..")
                    print("If you want to go back to the previous directory, type 'goback'.")
                    self.initiate_directory()

                else:
                    print("Invalid command, press enter to continue.")
                    continue

            else:
                if self.fileOverflow:
                    sys.stdout.write(CURSOR_UP_ONE)
                    sys.stdout.write(ERASE_LINE)
                print("There is no more files in the directory, please choose a file now: ")
                self.chosen_file = self.__parse_file(self.current_dir)
                break

    def oncyclekeypress(self,event):
        if event.name == "w":
            self.dirIndex+=1
            if self.dirIndex == len(self.dirs):
                self.dirIndex = 0
            sys.stdout.flush()
            sys.stdout.write(ERASE_LINE)
            print("\r\033[31;1;4m" + self.dirs[self.dirIndex] + "\033[0m", end="")
        if event.name == "enter":
            keyboard.unhook_all()

    def ynpress(self):
        while True:
            if keyboard.is_pressed("y") or keyboard.is_pressed("Y"):
                return True
            elif keyboard.is_pressed("n") or keyboard.is_pressed("N"):
                return False

    def __cycle_dirs(self):
        choosing_dir = True
        waitingForKey = True
        self.dirIndex = 0

        keyboard.on_press(self.oncyclekeypress)

        print("\033[31;1;4m" + self.dirs[self.dirIndex] + "\033[0m", end="")

        BUFFED=input()

        print("\n"+SEPERATOR)
        self.current_dir = os.path.expanduser(os.path.join(self.current_dir, self.dirs[self.dirIndex]))
        print("If you want to go back to the previous directory, type 'goback'.")
        self.initiate_directory()

    def ask_parameters(self):

        print("\n"+ SEPERATOR)
        if self.nii:
            print("**Warning: Spliting the axis of a dataset will make DTI Analysis incompatible.")
            time.sleep(0.1)
            with Waiting_Spinner(delay=0.15, askingText="Would you like to split the image by an axis? (y/n)"):
                self.process["Spliting"] = self.ynpress()
            print("\n")
            time.sleep(0.1)
            if self.process["Spliting"] == False:
                with Waiting_Spinner(delay=0.15, askingText="Would you like to perform DTI Analysis? (y/n)"):
                    self.process["DTI"] = self.ynpress()
                    time.sleep(0.1)
        else:
            print("This is a raw bruker image, to process it we must convert it to .nii.")
            time.sleep(0.1)
            with Waiting_Spinner(delay=0.15, askingText="Would you like to convert bruker to .nii? (y/n)"):
                self.process["Conversion"] = self.ynpress()
            print("\n")
            time.sleep(0.1)
            if self.process["Conversion"]:
                with Waiting_Spinner(delay=0.15, askingText="Would you like to split the image by an axis? (y/n)"):
                    self.process["Spliting"] = self.ynpress()
                print("\n")
                time.sleep(0.1)
                if self.process["Spliting"] == False:
                    with Waiting_Spinner(delay=0.15, askingText="Would you like to perform DTI Analysis? (y/n)"):
                        self.process["DTI"] = self.ynpress()
                    print("\n")
                    time.sleep(0.1)
                    if self.process["DTI"]:
                        with Waiting_Spinner(delay=0.15, askingText="Would you like FA file?"):
                            self.FA = self.ynpress()
                        print("\n")
                        time.sleep(0.1)
                        with Waiting_Spinner(delay=0.15, askingText="Would you like ADC file?"):
                            self.ADC = self.ynpress()
                        print("\n")
                        time.sleep(0.1)
                        with Waiting_Spinner(delay=0.15, askingText="Would you like lambda files?"):
                            self.LAMBDA = self.ynpress()
                        print("\n")
                        time.sleep(0.1)
                        with Waiting_Spinner(delay=0.15, askingText="Would you like RD file?"):
                            self.RD = self.ynpress()
                        print("\n")
                        time.sleep(0.1)


    def askForSplitAxis(self):
        if self.nii:
            img = nib.load(self.chosen_file)
        else:
            dataset = Dataset(self.chosen_file)
            img = nib.Nifti1Image(dataset.data, None)

        print("Here is the shape of your image, press enter to continue: " + str(img.shape), end="")

        block = input()

        while True:
            print("Pick an axis to split: ")
            raw = input()
            raw = raw.strip()
            d = [str(i) for i in list(img.shape)]
            if raw in d:
                self.splitIndex = raw
                break
            else:
                print("Invalid user input, try again please.")

    def get_firstfiles(self):
        while True:
            with Spinner(delay=0.15, askingText="Input your bruker/nifti file: "):
                #root.withdraw()
                self.chosen_file = filedialog.askopenfilename(title="Select nii/2dseq file")
                #root.destroy()
                if os.path.splitext(os.path.basename(self.chosen_file))[1] == ".nii":
                    print("Valid file")
                    self.nii = True
                    break
                else:
                    if os.path.basename(self.chosen_file) == "2dseq":
                        print("Valid file")
                        self.nii = False
                        break
                    else:
                        print("Invalid file, please try again."+"\n")
    
    def get_firstfilesarg(self):
        if os.path.exists(args.f):
            self.chosen_file = args.f
        else:
            print("INVALID IMAGE FILE")
            quit()

    def show_preview(self):
        print("\nHere's a preview...")
        self.qual = qualityChecker(self.chosen_file, nii=self.nii)
        self.s = self.qual.run()
        print("Finished quality_checker.")

    def format_paragraph(self):
        for idx, p in enumerate(self.process_text):
            print("\033[1;32m"+p+"\033[0m", end="")
            if idx < len(self.process_text)-1:
                print(ARROW, end="")

        print("\n")

    def print_pipeline(self):
        if self.process["Spliting"]:
            self.process_text.append(SPLITING)
        elif self.process["Conversion"]:
            self.process_text.append(CONVERSION)
            if self.process["Spliting"]:
                self.process_text.append(SPLITING)
            elif self.process["DTI"]:
                self.process_text.append(DIFFUSION)
                if self.process["Segmentation"]:
                    self.process_text.append(SEGMENTATION)
        elif self.process["DTI"]:
            self.process_text.append(DIFFUSION)
            if self.process["Segmentation"]:
                self.process_text.append(SEGMENTATION)

        self.process_text.append(SAVE_FILE)

        self.format_paragraph()
        with Waiting_Spinner(delay=0.15, askingText="Press enter to continue: "):
            if input():
                pass

    def askForOutputDir(self):
        with Spinner(delay=0.15, askingText="Input your output directory: "):
            self.OUTPUT_DIR = filedialog.askdirectory(title="Select output directory")
            print("\nSelected: " + self.OUTPUT_DIR)

    def askForNewFileName(self):
        print("Press enter to continue: ", end="")
        BUFFER = input()

        if not self.nii:
            print("Save under study name or pick a new name? (y/n): ")
            BUFF = self.parse_input()
            if BUFF:
                study_numb = os.path.abspath(os.path.join(os.path.join(os.path.join(self.chosen_file, ".."), ".."), ".."))
                subject = os.path.abspath(os.path.join(study_numb, ".."))
                self.new_name = os.path.basename(subject) + "_S" + os.path.basename(study_numb)
            else:
                print("Please type your new file name: ", end="")
                self.new_name = input()
        else:
            print("Please type your new file name: ", end="")
            self.new_name = input()

    def collect_data(self):
        subprocess.run("clear", shell=True)

        title = """
  ____ _____ ___   ____  _            _ _
 |  _ \_   _|_ _| |  _ \(_)_ __   ___| (_)_ __   ___
 | | | || |  | |  | |_) | | '_ \ / _ \ | | '_ \ / _ \ \n | |_| || |  | |  |  __/| | |_) |  __/ | | | | |  __/
 |____/ |_| |___| |_|   |_| .__/ \___|_|_|_| |_|\___|
                          |_|
 Created by Atul Phadke
        """

        description = """

    This pipline contains all of the following features,
    1. Conversion between BRUKER and NIFTI file types
    2. Saving 4D Images into a folder of 3D images
    3. DTI and generating FA, ADC, MD, etc.
    4. Atlas Segmentation and Statistics

        """

        print(title)
        print(description)

        print("We would now like to input the files, \n")

        with Waiting_Spinner(delay=0.15, askingText="Would you like to continue? Press enter to continue: "):
            if input() != "":
                quit()
            else:
                print(SEPERATOR+"\n")

        if args.cmd:
            self.get_firstfilescmd()
        else:
            #self.get_firstfilesarg()
            self.get_firstfiles()

        self.show_preview()
        time.sleep(0.4)

        self.ask_parameters()
        print("\n"+SEPERATOR)
        if self.process["Spliting"]:
            self.askForSplitAxis()
            print("\n"+SEPERATOR)
        self.askForOutputDir()
        print("\n"+SEPERATOR)
        self.askForNewFileName()

    def parse_input(self):
        while True:
            raw = input()
            raw = raw.strip()
            if raw in ["Y", "y", "n", "N"]:
                if raw in ["Y", "y"]:
                    return True
                else:
                    return False
            else:
                print("Invalid user input, try again please. \n")

    def run_pipeline(self):
        if self.process["Conversion"]:
            img, hdr = self.bruker2nifti()

            header = nib.Nifti1Header()
            #header.set_data_shape(hdr.shape[0:2])
            header["pixdim"] = hdr.resolution

            path = os.path.join(self.OUTPUT_DIR, (self.new_name+"_OG.nii"))
            nib.save(img, path)
            self.img = nib.load(path)
            if self.process["Spliting"]:
                self.splitImage()
            elif self.process["DTI"]:
                with Processing_Spinner(delay=0.15, askingText="Creating files.. "):
                    diff = DTI(self.chosen_file, self.new_name, self.OUTPUT_DIR, directions=self.s)
                    diff.generate_bvals()
                    tenfit = diff.dti_fit(self.img)
                    #b0 = nib.Nifti1Image(tenfit.evals, None).get_fdata()[:,:,:,0]
                    b0 = img.get_fdata()[:,:,:,0]
                    save_nifti(os.path.join(self.OUTPUT_DIR, (self.new_name+"_b0.nii")),nib.Nifti1Image(b0, None, header).get_fdata(), None)
                    if self.FA:
                        save_nifti(os.path.join(self.OUTPUT_DIR, (self.new_name+"_fa.nii")), nib.Nifti1Image(tenfit.fa, None, header).get_fdata(), None)
                    if self.ADC:
                        save_nifti(os.path.join(self.OUTPUT_DIR, (self.new_name+"_adc.nii")),nib.Nifti1Image(tenfit.adc, None, header).get_fdata(), None)
                    if self.LAMBDA:
                        l1, l2, l3 = dti._roll_evals(tenfit.evals, -1)
                        save_nifti(os.path.join(self.OUTPUT_DIR, (self.new_name+"_l1.nii")), nib.Nifti1Image(l1, None, header).get_fdata(), None)
                        save_nifti(os.path.join(self.OUTPUT_DIR, (self.new_name+"_l2.nii")), nib.Nifti1Image(l2, None, header).get_fdata(), None)
                        save_nifti(os.path.join(self.OUTPUT_DIR, (self.new_name+"_l3.nii")), nib.Nifti1Image(l3, None, header).get_fdata(), None)
                        
                    if self.RD:
                        save_nifti(os.path.join(self.OUTPUT_DIR, (self.new_name+"_rd.nii")), nib.Nifti1Image(tenfit.rd, None, header).get_fdata(), None)
                        #save_nifti(nib.Nifti1Image(b0, None, header).get_fdata(), os.path.join(self.OUTPUT_DIR, (self.new_name+"_rd.nii")))


                print("\nFinished.")
        else:
            self.img = nib.load(self.chosen_file)
            if self.process["Spliting"]:
                self.splitImage()
        if self.process["Spliting"]:
            self.splitImage()


    ## CONVERSION SOFTWARE
    def bruker2nifti(self):
        dataset = Dataset(self.chosen_file)
        return nib.Nifti1Image(dataset.data, None), dataset

    ##SPLITING SOFTWARE
    def splitImage(self):
        raw_img = self.img
        new_shape = list(raw_img.shape[::-1])
        index_slice = list(raw_img.shape[::-1]).index(int(self.splitIndex))
        new_shape[0], new_shape[index_slice] = new_shape[index_slice], new_shape[0]
        for idx, img in enumerate(np.reshape(np.transpose(np.rot90(raw_img.get_fdata())), new_shape)):
            output_image = nib.Nifti1Image(img, None)
            nib.save(output_image, os.path.join(self.OUTPUT_DIR, (self.new_name+str(idx)+".nii")))

    def run(self):
        self.collect_data()
        self.print_pipeline()
        self.run_pipeline()


class DTI:
    def __init__(self, DTI_FILE, NEW_NAME, OUTPUT_DIR, directions):
        self.chosen_file = DTI_FILE
        self.new_name = NEW_NAME
        self.OUTPUT_DIR = OUTPUT_DIR
        self.directions = directions

    def generate_bvals(self):
        method_file = os.path.abspath(os.path.join(os.path.join(os.path.join(self.chosen_file, ".."), ".."), "..")) + "/method"
        f=open(method_file)
        no_line_breaks = f.read()
        content = no_line_breaks.split("\n")

        bval = None
        dwDir = None
        GradOrient = None

        for line in content:
            if "PVM_DwBvalEach" in line and not bval:
                bval = content[content.index(line)+1]

            elif "PVM_SPackArrGradOrient" in line and not GradOrient:

                reshape = line.replace("##$PVM_SPackArrGradOrient=( ", "")
                reshape = reshape.replace(" )", "").replace(",", "")
                reshape = list(reshape.split(" "))
                reshape = tuple([int(item) for item in reshape])

                vals = np.prod(list(reshape))

                GradOrientArray = content[content.index(line)+1:content.index(line)+4]
                GradOrient = []

                for c in GradOrientArray:
                    d = c.split(" ")
                    for grd in d:
                        GradOrient.append(grd)

                GradOrient = np.array(list(filter(None, GradOrient)))
                GradOrient = GradOrient[0:vals]
                GradOrient.shape = reshape
                GradOrient = np.squeeze(GradOrient)
                GradOrient = GradOrient.astype(float)

            elif "##$PVM_DwDir=" in line and not dwDir:

                dwDirArray = no_line_breaks[no_line_breaks.index(content[content.index(line)+1]):no_line_breaks.find("#", no_line_breaks.index(content[content.index(line)+2]))].split(" ")
                dwDir = [0,0,0]

                for idx, element in enumerate(dwDirArray):
                    f = math.floor(idx/3) + 1
                    if self.directions[("b"+str(f))]:
                        dwDir.append(element.strip())

                dwDir = np.array(dwDir)
                dwDir.shape = (int(len(dwDir)/3),3)
                dwDir = dwDir.astype(float)
 
        bvec = np.dot(dwDir, GradOrient)
        bvec_file = open(os.path.join(self.OUTPUT_DIR, self.new_name+".bvec"), "w+")
        bval_file = open(os.path.join(self.OUTPUT_DIR, self.new_name+".bval"), "w+")

        bval_file.truncate()

        numb = sum(value == True for value in self.directions.values())

        bval_file.write("0 " + (len(dwDir)-1)*(str(bval) + " "))
        bval_file.close()

        bvec_file.truncate()

        bvec.shape = (len(dwDir), 3)

        for vector_array in bvec:
            for vector in vector_array:
                bvec_file.write(str(vector) + " ")
            bvec_file.write("\n")

        bvec_file.close()

    def dti_fit(self, img):
        fbval = os.path.join(self.OUTPUT_DIR, self.new_name+".bval")
        fbvec = os.path.join(self.OUTPUT_DIR, self.new_name+".bvec")

        bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
        gtab = gradient_table(bvals, bvecs)
        tenmodel = TensorModel(gtab)

        fdata = np.array(img.get_fdata())
        for x in range(0, fdata.shape[-1]):
            if not self.directions["b"+str(x)]:
                fdata = np.delete(fdata, x, -1)


        tenfit = tenmodel.fit(fdata)

        return tenfit

class Spinner:
    busy = False
    delay = 0.1

    @staticmethod
    def spinning_cursor(animation):
        while 1:
            for cursor in animation: yield cursor

    def __init__(self, delay=None, askingText=""):
        animation = ["⢿", "⢿", "⣻", "⣻", "⣽", "⣽", "⣾", "⣾", "⣷", "⣷", "⣯", "⣯", "⣟", "⣟", "⡿", "⡿"]
        self.spinner_generator = self.spinning_cursor(animation)
        if delay and float(delay):
            self.delay = delay
            self.askingText = askingText

    def spinner_task(self):
        while self.busy:
            sys.stdout.flush()
            sys.stdout.write(self.askingText + " " + next(self.spinner_generator) + "\n")
            sys.stdout.flush()
            time.sleep(self.delay)
            sys.stdout.write("\033[F")
            sys.stdout.write('\b'*len(self.askingText + " " + next(self.spinner_generator) + "\n"))
            sys.stdout.flush()

    def __enter__(self):
        self.busy = True
        sys.stdout.flush()
        threading.Thread(target=self.spinner_task).start()

    def __exit__(self, exception, value, tb):
        self.busy = False
        time.sleep(self.delay)
        if exception is not None:
            return False

class qualityChecker:
    def __init__(self, img, nii):
        self.axarr0Click = False
        self.axarr1Click = False
        self.axarr2Click = False

        if not nii:
            self.img = Dataset(img).data
            nib.save(nib.Nifti1Image(self.img, None), os.path.join(os.getcwd(),"cache", "temp.nii"))
            sitk_t1 = sitk.ReadImage(os.path.join(os.getcwd(),"cache", "temp.nii"))
            self.img = sitk.GetArrayFromImage(sitk_t1)
        else:
            sitk_t1 = sitk.ReadImage(img)
            self.img = sitk.GetArrayFromImage(sitk_t1)

        self.AXIS2 = self.img.shape[-1]
        self.AXIS0 = self.img.shape[-2]
        self.AXIS1 = self.img.shape[-3]

        self.DIRECTION = 0

        self.f, self.axarr = plt.subplots(1, 3, figsize=(10,5))
        self.f.suptitle("B"+str(self.DIRECTION)+" Image", fontsize=15, fontweight="bold")
        self.axarr[1].set_title("Coronal", fontsize=12)
        self.axarr[0].set_title("Axial", fontsize=12)
        self.axarr[2].set_title("Saggital", fontsize=12)


        self.CURRENT1 = round(self.AXIS1/2)
        self.CURRENT0 = round(self.AXIS0/2)
        self.CURRENT2 = round(self.AXIS2/2)
        
        self.axarr[1].set_xlabel(str(self.CURRENT1)+"/"+str(self.AXIS1))
        self.axarr[0].set_xlabel(str(self.CURRENT0)+"/"+str(self.AXIS0))
        self.axarr[2].set_xlabel(str(self.CURRENT2)+"/"+str(self.AXIS2))

        self.img0 = self.axarr[0].imshow(self.img[self.DIRECTION,:,self.CURRENT0,:], cmap='gray')
        self.img1 = self.axarr[1].imshow(self.img[self.DIRECTION,self.CURRENT1,:,:], cmap='gray')
        self.img2 = self.axarr[2].imshow(self.img[self.DIRECTION,:,:,self.CURRENT2], cmap='gray')

        self.axprev = self.f.add_axes([0.7, 0.05, 0.1, 0.075])
        self.axnext = self.f.add_axes([0.81, 0.05, 0.1, 0.075])
        self.axfinish = self.f.add_axes([0.05, 0.05, 0.1, 0.075])
        #self.ax_elim = self.f.add_axes([0.412, 0.85, 0.2, 0.04])
        self.ax_check = self.f.add_axes([0.01, 0.75, 0.15, 0.2])

        self.ax0background = self.f.canvas.copy_from_bbox(self.axarr[0].bbox)
        self.axbackground = self.f.canvas.copy_from_bbox(self.axarr[1].bbox)
        self.ax2background = self.f.canvas.copy_from_bbox(self.axarr[2].bbox)

        self.eliminate = None
        self.check_status = []
        self.b_images = []
        self.vis = []
        for x in range(0, self.img.shape[0]):
            self.b_images.append("b"+str(x))
            self.vis.append(True)

        self.directions = {}

        for b in self.b_images:
            self.directions[str(b)] = True

        self.ax_check.set_visible(False)

        self.check = CheckButtons(self.ax_check, ["Keep Direction"], [True])
        self.check.on_clicked(self.func)

    def onclick_select(self, event):
        if event.inaxes == self.axarr[1]:
            if self.axarr1Click:
                self.axarr1Click = False
            else:
                self.axarr1Click = True
                self.axarr0Click = False
                self.axarr2Click = False
        elif event.inaxes == self.axarr[0]:
            if self.axarr0Click:
                self.axarr0Click = False
            else:
                self.axarr0Click = True
                self.axarr1Click = False
                self.axarr2Click = False
        elif event.inaxes == self.axarr[2]:
            if self.axarr2Click:
                self.axarr2Click = False
            else:
                self.axarr2Click = True
                self.axarr0Click = False
                self.axarr1Click = False
        elif event.inaxes == self.axnext:
            if self.DIRECTION < (self.img.shape[0]-1):
                self.DIRECTION +=1
                self.img0.set_data(self.img[self.DIRECTION, :,self.CURRENT0,:])
                self.img1.set_data(self.img[self.DIRECTION,self.CURRENT1, :, :])
                self.img2.set_data(self.img[self.DIRECTION,:,:,self.CURRENT2])
                self.f.suptitle("B"+str(self.DIRECTION)+" Image", fontsize=15, fontweight="bold")
                #print(self.directions["b"+str(self.DIRECTION)])
                if self.directions["b"+str(self.DIRECTION)] and not self.check.lines[0][0].get_visible():
                    self.check.set_active(0)

                if not self.directions["b"+str(self.DIRECTION)] and self.check.lines[0][0].get_visible():
                    self.check.set_active(0)

                if self.DIRECTION != 0:
                    self.ax_check.set_visible(True)
                else:
                    self.ax_check.set_visible(False)

                self.f.canvas.draw_idle()

        elif event.inaxes == self.axprev:
            if self.DIRECTION > 0:
                self.DIRECTION -=1
                self.img0.set_data(self.img[self.DIRECTION, :,self.CURRENT0,:])
                self.img1.set_data(self.img[self.DIRECTION,self.CURRENT1, :, :])
                self.img2.set_data(self.img[self.DIRECTION,:,:,self.CURRENT2])
                self.f.suptitle("B"+str(self.DIRECTION)+" Image", fontsize=15, fontweight="bold")
                
                if self.directions["b"+str(self.DIRECTION)] and not self.check.lines[0][0].get_visible():
                    self.check.set_active(0)

                if not self.directions["b"+str(self.DIRECTION)] and self.check.lines[0][0].get_visible():
                    self.check.set_active(0)

                if self.DIRECTION != 0:
                    self.ax_check.set_visible(True)
                else:
                    self.ax_check.set_visible(False)

                self.f.canvas.draw_idle()

        elif event.inaxes == self.axfinish:
            plt.close()

    def func(self,label):
        #print(self.check.lines[0][0].get_visible())
        if self.check.lines[0][0].get_visible():
            self.directions["b"+str(self.DIRECTION)] = True
        else:
            self.directions["b"+str(self.DIRECTION)] = False

    def mouse_move(self, event):
        if event.inaxes == self.axarr[1]:
            if self.axarr1Click:
                x, y = round(event.xdata), round(event.ydata)
                if x > self.img.shape[-2]:
                    x = self.img.shape[-2]
                if y > self.img.shape[-1]:
                    y = self.img.shape[-1]
                self.img0.set_data(self.img[self.DIRECTION,:,x,:])
                self.img2.set_data(self.img[self.DIRECTION,:,:,y])

                self.axarr[0].set_xlabel(str(x)+"/"+str(self.AXIS0))
                self.axarr[2].set_xlabel(str(y)+"/"+str(self.AXIS2))
                self.CURRENT0 = x
                self.CURRENT2 = y
                self.f.canvas.draw_idle()
                self.f.canvas.flush_events()
                plt.pause(0.000001)
        elif event.inaxes == self.axarr[0]:
            if self.axarr0Click:
                x, y = round(event.xdata), round(event.ydata)
                if x > self.img.shape[-1]:
                    x = self.img.shape[-1]
                if y > self.img.shape[1]:
                    y = self.img.shape[1]

                self.img1.set_data(self.img[self.DIRECTION,y,:,:])
                self.img2.set_data(self.img[self.DIRECTION,:,:,x])
                self.axarr[1].set_xlabel(str(y)+"/"+str(self.AXIS1))
                self.axarr[2].set_xlabel(str(x)+"/"+str(self.AXIS2))
                self.CURRENT1 = y
                self.CURRENT2 = x
                self.f.canvas.draw_idle()
                self.f.canvas.flush_events()
                plt.pause(0.000001)
        elif event.inaxes == self.axarr[2]:
            if self.axarr2Click:
                x, y = round(event.xdata), round(event.ydata)
                if x > self.img.shape[-2]:
                    x = self.img.shape[-2]
                if y > self.img.shape[1]:
                    y = self.img.shape[1]
                self.img0.set_data(self.img[self.DIRECTION,:,x,:])
                self.img1.set_data(self.img[self.DIRECTION,y,:,:])
                self.axarr[0].set_xlabel(str(x)+"/"+str(self.AXIS0))
                self.axarr[1].set_xlabel(str(y)+"/"+str(self.AXIS1))
                self.CURRENT0 = x
                self.CURRENT1 = y
                self.f.canvas.draw_idle()
                self.f.canvas.flush_events()
                plt.pause(0.000001)

    def run(self):

        bnext = Button(self.axnext, 'Next')

        bprev = Button(self.axprev, 'Previous')

        finish = Button(self.axfinish, "Finish")

        directions = self.img.shape[0]
        
        self.f.canvas.mpl_connect("button_press_event",self.onclick_select)
        self.f.canvas.mpl_connect("motion_notify_event",self.mouse_move)

        plt.show()

        return self.directions

class Processing_Spinner(Spinner):
    busy = False
    delay = 0.3

    def __init__(self, delay=None, askingText=""):
        animation = [
        "[        ]", "[        ]",
        "[=       ]", "[=       ]",
        "[==      ]", "[==      ]",
        "[===     ]", "[===     ]",
        "[====    ]", "[====    ]",
        "[=====   ]", "[=====   ]",
        "[======  ]", "[======  ]",
        "[======= ]", "[======= ]",
        "[========]", "[========]",
        "[ =======]", "[ =======]",
        "[  ======]", "[  ======]",
        "[   =====]", "[   =====]",
        "[    ====]", "[    ====]",
        "[     ===]", "[     ===]",
        "[      ==]", "[      ==]",
        "[       =]", "[       =]",
        "[        ]", "[        ]",
        "[        ]", "[        ]"]
        self.spinner_generator = self.spinning_cursor(animation)
        if delay and float(delay):
            self.delay = delay
            self.askingText = askingText

class Waiting_Spinner(Spinner):
    busy = False
    delay = 0.3

    def __init__(self, delay=None, askingText=""):
        animation = ["◜", "◜", "◝", "◝", "◞", "◞", "◟", "◟"]
        self.spinner_generator = self.spinning_cursor(animation)
        if delay and float(delay):
            self.delay = delay
            self.askingText = askingText

class yn_spinner(Spinner):
    busy = False
    delay = 0.3

    def __init__(self, delay=None, askingText=""):
        self.askingText = askingText
        self.animation = ["◜", "◜", "◝", "◝", "◞", "◞", "◟", "◟"]
        self.spinner_generator = self.spinning_cursor(self.animation)
        if delay and float(delay):
            self.delay = delay

    def __enter__(self):
        self.busy = True
        threading.Thread(target=self.spinner_task).start()
        threading.Thread(target=self.catch_inputs).start()

    def catch_inputs(self):
        while True:
            if keyboard.is_pressed("y"):
                print("y\n")
                break

instance = Pipeline()

instance.run()
