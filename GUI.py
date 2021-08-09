import requests
import json
from tkinter import *
from tkinter import ttk
from PIL import ImageTk,Image
from tkinter import messagebox
from tkinter import filedialog
import server
from flask import Flask,jsonify,request
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from matplotlib import pyplot as plt
from PIL import ImageTk,Image

def create_app():
    app = Flask(__name__)

    with app.app_context():
        server.get_location_structure()

    return app

try:
    create_app()
    url = "http://127.0.0.1:5000/get_location_structure"
    response = requests.get( url, headers={'Accept': "application/json"})
except Exception as e:
    print(e)


class UI:
    def __init__(self,master):
        self.master = master
        self.master.title("SaveLife")
        self.master.resizable(False, False)
        # resizing the window created
        self.screen_width = self.master.winfo_screenwidth()
        self.screen_height = self.master.winfo_screenheight()
        window_height = 900
        window_width = 1200
        x_cordinate = int((self.screen_width / 2) - (window_width / 2))
        y_cordinate = int((self.screen_height / 2) - (window_height / 2))

        self.master.geometry("{}x{}+{}+{}".format(int(window_width), int(window_height), x_cordinate, y_cordinate))
        self.master.configure(bg='white')
        self.master.attributes('-alpha', 0.0)
        self.master.after(0, self.master.attributes, "-alpha", 1.0)

        self.banner = Label(self.master, text="SaveLife", font="Helvetica 18 bold", bg="red", width=100)
        self.banner.place(x=0, y=6)

        self.my_img_icon_file = ImageTk.PhotoImage(Image.open("home.png"))

        self.label = Label(self.master, image=self.my_img_icon_file)
        self.label.place(x=10,y=0)
        self.label.image = self.my_img_icon_file

        self.post = Label(self.master, text="SOLAR PV OUTPUT CALCULATOR", font="Helvetica 18 bold", width=50, bg="white")
        self.post.place(x=0, y=40)

        self.finger_print_recognition_button = Button(self.master, text="Calculator", width=30, command = self.calculator_clicked)
        self.finger_print_recognition_button.place(x=5, y=110)

        self.search_button = Button(self.master, text="Graphs", width=30, command = self.graph_clicked)
        self.search_button.place(x=5, y=140)

        self.about_button = Button(self.master, text="About", width=30)
        self.about_button.place(x=5, y=170)

        self.frame = Frame(self.master, bg="white")
        self.frame.place(x=260, y=80)

        self.temperature_label = Label(self.master, text="Temperature", font = "Helvetica 12")
        self.temperature_label.place(x=270, y=110)

        self.relative_humidity_2_m_above_gnd = Label(self.master, text="Relative humidity", font = "Helvetica 12")
        self.relative_humidity_2_m_above_gnd.place(x=270, y=150)

        self.mean_sea_level_pressure_MSL = Label(self.master, text="Mean sea level pressure", font="Helvetica 12")
        self.mean_sea_level_pressure_MSL.place(x=270, y=190)

        self.wind_direction_10_m_above_gnd = Label(self.master, text="Wind direction (10m)", font="Helvetica 12")
        self.wind_direction_10_m_above_gnd.place(x=270, y=230)

        self.wind_speed_10_m_above_gnd = Label(self.master, text="Wind speed (10m)", font="Helvetica 12")
        self.wind_speed_10_m_above_gnd.place(x=270, y=270)

        self.zenith = Label(self.master, text="Zenith", font="Helvetica 12")
        self.zenith.place(x=270, y=310)

        self.azimuth = Label(self.master, text="Azimuth", font="Helvetica 12")
        self.azimuth.place(x=270, y=350)

        self.temperature_label_entry = Entry(self.master)
        self.temperature_label_entry.place(x=500, y=110)

        self.relative_humidity_2_m_above_gnd_entry = Entry(self.master)
        self.relative_humidity_2_m_above_gnd_entry.place(x=500,y=150)

        self.mean_sea_level_pressure_MSL_entry = Entry(self.master)
        self.mean_sea_level_pressure_MSL_entry.place(x=500, y=190)

        self.wind_direction_10_m_above_gnd_entry = Entry(self.master)
        self.wind_direction_10_m_above_gnd_entry.place(x=500, y=230)

        self.wind_speed_10_m_above_gnd_entry = Entry(self.master)
        self.wind_speed_10_m_above_gnd_entry.place(x=500, y=270)

        self.zenith_entry = Entry(self.master)
        self.zenith_entry.place(x=500, y=310)

        self.azimuth_entry = Entry(self.master)
        self.azimuth_entry.place(x=500, y=350)

        self.calculate = Button(self.master,text="Calculate",command = self.calculate)
        self.calculate.place(x=500,y=390)

        self.DHI = LabelFrame(self.master,text='DHI (W/m2)',font="Helvetica 12")
        self.DHI.place(x=270,y=440,height=45)
        self.dhi_label = Text(self.DHI, width=28)
        self.dhi_label.insert(END, str(0.00))
        self.dhi_label.configure(state='disabled')
        self.dhi_label.pack()

        self.GHI = LabelFrame(self.master, text='GHI (W/m2)', font="Helvetica 12")
        self.GHI.place(x=270, y=500, height=45)
        self.ghi_label = Text(self.GHI, width=28)
        self.ghi_label.insert(END, str(0.00))
        self.ghi_label.configure(state='disabled')
        self.ghi_label.pack()

        self.power_output = LabelFrame(self.master, text='Generated power (KW)', font="Helvetica 12")
        self.power_output.place(x=270, y=560, height=45)
        self.power_output_label = Text(self.power_output, width=28)
        self.power_output_label.insert(END, str(0.00))
        self.power_output_label.configure(state='disabled')
        self.power_output_label.pack()

    def calculate(self):
        ghi_value = server.get_estimated_GHI(float(self.temperature_label_entry.get()),float(self.relative_humidity_2_m_above_gnd_entry.get()),
                                 float(self.mean_sea_level_pressure_MSL_entry.get()),float(self.wind_direction_10_m_above_gnd_entry.get()),
                                 float(self.wind_speed_10_m_above_gnd_entry.get()),float(self.zenith_entry.get()))
        dhi_value = server.get_estimated_DHI(float(self.temperature_label_entry.get()),float(self.relative_humidity_2_m_above_gnd_entry.get()),
                                 float(self.mean_sea_level_pressure_MSL_entry.get()),float(self.wind_direction_10_m_above_gnd_entry.get()),
                                 float(self.wind_speed_10_m_above_gnd_entry.get()),float(self.zenith_entry.get()))

        solar_output = server.get_estimated_power_output_kw(float(self.temperature_label_entry.get()),float(self.relative_humidity_2_m_above_gnd_entry.get()),
                                 float(self.mean_sea_level_pressure_MSL_entry.get()),float(self.wind_direction_10_m_above_gnd_entry.get()),
                                 float(self.wind_speed_10_m_above_gnd_entry.get()),float(self.zenith_entry.get()),float(self.azimuth_entry.get()),dhi_value,ghi_value)
        #print(solar_output[0][0])
        self.power_output_label.configure(state='normal')
        self.power_output_label.delete(1.0, 'end')
        self.power_output_label.insert(1.0, str(solar_output[0][0]))
        self.power_output_label.configure(state='disabled')

        self.ghi_label.configure(state='normal')
        self.ghi_label.delete(1.0, 'end')
        self.ghi_label.insert(1.0, str(ghi_value[0]))
        self.ghi_label.configure(state='disabled')

        self.dhi_label.configure(state='normal')
        self.dhi_label.delete(1.0, 'end')
        self.dhi_label.insert(1.0, str(dhi_value[0]))
        self.dhi_label.configure(state='disabled')



    def graph_clicked(self):

        self.small_win = Toplevel()
        self.small_win.resizable(False, False)
        # resizing the window created
        window_height = 900
        window_width = 1100
        self.small_win.geometry("{}x{}".format(window_width, window_height))

        df = pd.read_csv('seen.csv')
        predicted_DHI = pd.read_csv("DHI_predicted.csv")
        predicted_GHI = pd.read_csv("GHI_predicted.csv")
        print(len(predicted_DHI), len(df))
        df = pd.concat([df, predicted_DHI, predicted_GHI], axis='columns')
        df = df.dropna()
        self.df = df.drop(
            ['total_precipitation_sfc', 'snowfall_amount_sfc', 'total_cloud_cover_sfc', 'high_cloud_cover_high_cld_lay',
             'medium_cloud_cover_mid_cld_lay', 'low_cloud_cover_low_cld_lay', 'shortwave_radiation_backwards_sfc',
             'wind_speed_80_m_above_gnd', 'wind_direction_80_m_above_gnd', 'wind_speed_900_mb', 'wind_direction_900_mb',
             'wind_gust_10_m_above_gnd', 'angle_of_incidence'], axis='columns')
        self.temp_value = StringVar()
        self.temp_value.set("no")

        self.hist_plot(self.df,'temperature_2_m_above_gnd',10,10)

        self.temp_b = Checkbutton(self.small_win,text="Temperature",variable=self.temp_value, onvalue='temperature_2_m_above_gnd', offvalue="no")
        self.temp_b.place(x=15,y=260)

        self.humidity_value = StringVar()
        self.humidity_value.set("no")

        self.hist_plot(self.df, 'relative_humidity_2_m_above_gnd', 10, 280)

        self.humidity_b = Checkbutton(self.small_win, text="Relative humidity", variable=self.humidity_value,
                                  onvalue='relative_humidity_2_m_above_gnd', offvalue="no")
        self.humidity_b.place(x=15, y=530)

        self.pressure_value = StringVar()
        self.pressure_value.set("no")

        self.hist_plot(self.df, 'mean_sea_level_pressure_MSL', 10, 550)

        self.pressure_b = Checkbutton(self.small_win, text="Mean sea level pressure", variable=self.pressure_value,
                                      onvalue='mean_sea_level_pressure_MSL', offvalue="no")
        self.pressure_b.place(x=15, y=800)

        self.win_dir_value = StringVar()
        self.win_dir_value.set("no")

        self.hist_plot(self.df, 'wind_direction_10_m_above_gnd', 270, 10)

        self.win_dir_b = Checkbutton(self.small_win, text="Wind direction", variable=self.win_dir_value,
                                      onvalue='wind_direction_10_m_above_gnd', offvalue="no")
        self.win_dir_b.place(x=275, y=260)

        self.win_speed_value = StringVar()
        self.win_speed_value.set("no")

        self.hist_plot(self.df, 'wind_speed_10_m_above_gnd', 270, 280)

        self.win_speed_b = Checkbutton(self.small_win, text="Wind speed", variable=self.win_speed_value,
                                     onvalue='wind_speed_10_m_above_gnd', offvalue="no")
        self.win_speed_b.place(x=275, y=530)

        self.zenith_value = StringVar()
        self.zenith_value.set("no")

        self.hist_plot(self.df, 'zenith', 270, 550)

        self.zenith_b = Checkbutton(self.small_win, text="Zenith", variable=self.zenith_value,
                                       onvalue='zenith', offvalue="no")
        self.zenith_b.place(x=275, y=800)

        self.azimuth_value = StringVar()
        self.azimuth_value.set("no")

        self.hist_plot(self.df, 'azimuth', 530, 10)

        self.azimuth_b = Checkbutton(self.small_win, text="Azimuth", variable=self.azimuth_value,
                                    onvalue='azimuth', offvalue="no")
        self.azimuth_b.place(x=535, y=260)

        self.dhi_value = StringVar()
        self.dhi_value.set("no")

        self.hist_plot(self.df, 'DHI', 530, 280)

        self.dhi_b = Checkbutton(self.small_win, text="Dhi", variable=self.dhi_value,
                                     onvalue='DHI', offvalue="no")
        self.dhi_b.place(x=535, y=530)

        self.hist_plot(self.df, 'GHI', 530, 550)

        self.ghi_value = StringVar()
        self.ghi_value.set("no")

        self.ghi_b = Checkbutton(self.small_win, text="Ghi", variable=self.ghi_value,
                                 onvalue='GHI', offvalue="no")
        self.ghi_b.place(x=535, y=800)

        self.power_value = StringVar()
        self.power_value.set("no")

        self.hist_plot(self.df, 'generated_power_kw', 790, 10)

        self.power_b = Checkbutton(self.small_win, text="Power genrated", variable=self.power_value,
                                 onvalue='generated_power_kw', offvalue="no")
        self.power_b.place(x=795, y=260)

        self.plot_button = Button(self.small_win, text='Plot',bg='red',command = self.plot_button_clicked)
        self.plot_button.place(x=850,y= 800)

        # binding submit button
        self.plot_button.bind("<Enter>", self.cursor_enter_plot)
        self.plot_button.bind("<Leave>", self.cursor_leave_plot)

    def plot_button_clicked(self):
        self.plot_relationship(self.df, self.plot_value[0], self.plot_value[1])

    def cursor_enter_plot(self,event):
        list = []
        self.plot_value = []
        i = 0
        list.append(self.temp_value.get())
        list.append(self.humidity_value.get())
        list.append(self.pressure_value.get())
        list.append(self.win_dir_value.get())
        list.append(self.win_speed_value.get())
        list.append(self.zenith_value.get())
        list.append(self.azimuth_value.get())
        list.append(self.dhi_value.get())
        list.append(self.ghi_value.get())
        list.append(self.power_value.get())
        for item in list:
            if item=='no':
                i+=1
            else:
                pass
        if i == 8:
            for l in list:
                if l !='no':
                    self.plot_value.append(l)
                else:
                    pass


        else:
            self.plot_button.config(state=DISABLED)

    def plot_relationship(self,df,x,y):
        plt.scatter(df[x], df[y], marker='+')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

    def cursor_leave_plot(self,event):
        self.plot_button.config(state=NORMAL)

    def check_button_clicked(self,value):
        print(value)


    def hist_plot(self,df,name,x,y):

        plt.hist(df[name], bins=50, rwidth=0.8)
        plt.savefig("virtual_img.PNG")
        image_9 = ImageTk.PhotoImage((Image.open("virtual_img.PNG")).resize((250, 250), Image.ANTIALIAS))
        self.presure_label = Label(self.small_win, image=image_9)
        self.presure_label.place(x=x, y=y)
        self.presure_label.image = image_9
        plt.clf()


    def calculator_clicked(self):
        remove = self.frame.winfo_children()
        for child in remove:
            child.destroy()

        #'temperature_2_m_above_gnd', 'relative_humidity_2_m_above_gnd',
        #'mean_sea_level_pressure_MSL', 'wind_direction_10_m_above_gnd',
        #'wind_speed_10_m_above_gnd', 'zenith', 'azimuth', 'generated_power_kw',
        #'DHI', 'GHI'

        #'Temperature(deg.C)', 'Relative humidity(%)', 'Pressure(mbar)',
        #'Wind Direction(degree)', 'Wind Speed(m/s)', 'Zenith(degree)'


if __name__ == '__main__' :
    root = Tk()
    UI(root)
    root.mainloop()


