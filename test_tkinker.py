import tkinter as tk

'''  Manually Folders Listing  '''
folder_name_list = [
    '1WayConnectorforFoley',
    '2WayConnectorforFoley',
    '2WayFoleyCatheter',
    '3WayConnectorforFoley',
    '3Waystopcock',
    'AlcoholBottle',
    'AlcoholPad',
    'CottonBall',
    'CottonSwap',
    'Dilator',
    'DisposableInfusionSet',
    'ExtensionTube',
    'FaceShield',
    'FootWear',
    'FrontLoadSyringe',
    'GauzePad',
    'Glove',
    'GuideWire',
    'LiquidBottle',
    'Mask',
    'NasalCannula',
    'Needle',
    'NGTube',
    'OxygenMask',
    'PharmaceuticalProduct',
    'Pill',
    'PillBottle',
    'PPESuit',
    'PrefilledHumidifier',
    'PressureConnectingTube',
    'ReusableHumidifier',
    'SodiumChlorideBag',
    'SterileHumidifierAdapter',
    'SurgicalBlade',
    'SurgicalCap',
    'SurgicalSuit',
    'Syringe',
    'TrachealTube',
    'UrineBag',
    'Vaccinebottle',
    'WingedInfusionSet',
]


changedLabel = None


class SimpleSelectLabel():
    def __init__(self,labelList,nCol=6):
        numInRow = nCol
        self.window = tk.Tk()
        self.window.geometry("1800x900")
        i = 0 # row_index
        iLabel = 0 # index of Label
        nLabel = len(folder_name_list)
        while(1):
            self.window.columnconfigure(i, weight=1, minsize=90)
            self.window.rowconfigure(i, weight=1, minsize=50)
            for j in range(0, numInRow):
                frame = tk.Frame(
                    master=self.window,
                    relief=tk.RAISED,
                    borderwidth=1
                )
                frame.grid(row=i, column=j, padx=10, pady=10)
                label = tk.Label(master=frame, text=labelList[iLabel])
                label.config(font=('Helvatical bold',18))
                txtLabel = labelList[iLabel]
                label.bind("<Button-1>",lambda event,text=txtLabel:self.hLabelClick(event,text))
                iLabel+=1 # fecth next LabelS
                label.pack(padx=3, pady=3)
                if(iLabel>=nLabel):
                    break
            i+=1 # new row
            if(iLabel>=nLabel):
                break
        self.window.mainloop()
    def hLabelClick(self,event,text):
        global changedLabel
        changedLabel = text
        self.window.quit()

SimpleSelectLabel(folder_name_list)
print(changedLabel)
