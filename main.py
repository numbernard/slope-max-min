import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title('Excel File Viewer')

# Add a file uploader to the app
file = st.file_uploader('Upload an Excel file', type=['xls', 'xlsx'])
def is_point_in_line_bounds(x, y, slope, intercept, yerror):
    lower_bound = y - yerror
    upper_bound = y + yerror
    return lower_bound <= (slope * x + intercept) <= upper_bound

#slope max
def find_max_slope(data, yerrors, max_iterations):
    # Check the slope for the first and last points
    n = len(data)
    first_point = [data[0, 0], data[0, 1] - yerrors[0]]
    last_point = [data[n-1, 0], data[n-1, 1] + yerrors[n-1]]
    line_data = np.array([first_point, last_point])
    slope, intercept, r_value, p_value, std_err = linregress(line_data[:,0], line_data[:,1])
    # Check if all points lie within the bounds of the line for the first slope
    if all([is_point_in_line_bounds(data[i, 0], data[i, 1], slope, intercept, yerrors[i]) for i in range(n)]):
        return True, True, slope, intercept
    # If not, iterate until the slope of the line stops changing or the maximum number of iterations is reached
    d2 = 0
    n_drop = np.zeros(n)
    itera=0
    while itera<=max_iterations:
        d1 = 0.001
        while (data[0, 1] - yerrors[0] + d1) <= (data[0, 1] + yerrors[0]):
            # Define the endpoints of the line based on the current values of d1 and d2
            line_start = [data[0, 0], data[0, 1] - yerrors[0] + d1]
            line_end = [data[n-1, 0], data[n-1, 1] + yerrors[n-1] - d2]
            line_data = np.array([line_start, line_end])
            # Compute the slope, intercept, and correlation coefficient of the line
            slope, intercept, r_value, p_value, std_err  = linregress(line_data[:,0], line_data[:,1])
            # Check which points lie within the bounds of the line and count how many there are
            num_points_within_bounds = sum([is_point_in_line_bounds(data[i, 0], data[i, 1], slope, intercept, yerrors[i]) for i in range(n)])
            # If all points lie within the bounds of the line, we're done
            if num_points_within_bounds == n:
                return True, True, slope, intercept
            # If not, mark the points that are outside the bounds of the line for dropping
            else:
                for i in range(n):
                    if not is_point_in_line_bounds(data[i, 0], data[i, 1], slope, intercept, yerrors[i]):
                        n_drop[i] += 1
                    itera+=1
            d1 += 0.001
        # If we've reached the end of the line, move the line up
        if (data[n-1, 1] + yerrors[n-1] - d2) >= (data[n-1, 1] - yerrors[n-1]):
            d2 += 0.001
        # If we've reached the end of the data, drop the point with the most marks and try again
        else:
            print("The condition is fail")
            print("The following points were dropped:")
            m = np.max(n_drop)
            for i in range(n):
                if n_drop[i] == m:
                    print(f"({data[i,0]}, {data[i,1]})")
                    data = np.delete(data, i, axis=0)
                    yerrors = np.delete(yerrors, i)
                    n=len(data)
                    return data, yerrors, False, False # break out of the loop after dropping the first point with the maximum number of marks
    #return data, yerrors, slope, intercept

#slope min
def find_min_slope(data, yerrors, max_iterations):
    # Check the slope for the first and last points
    n = len(data)
    first_point = [data[0, 0], data[0, 1] + yerrors[0]]
    last_point = [data[n-1, 0], data[n-1, 1] - yerrors[n-1]]
    line_data = np.array([first_point, last_point])
    slope, intercept, r_value, p_value, std_err = linregress(line_data[:,0], line_data[:,1])
    # Check if all points lie within the bounds of the line for the first slope
    if all([is_point_in_line_bounds(data[i, 0], data[i, 1], slope, intercept, yerrors[i]) for i in range(n)]):
        return True, True, slope, intercept
    # If not, iterate until the slope of the line stops changing or the maximum number of iterations is reached
    d2 = 0
    n_drop = np.zeros(n)
    itera=0
    while itera<=max_iterations:
        d1 = 0.001
        while (data[0, 1] + yerrors[0] - d1) >= (data[0, 1] - yerrors[0]):
            # Define the endpoints of the line based on the current values of d1 and d2
            line_start = [data[0, 0], data[0, 1] + yerrors[0] - d1]
            line_end = [data[n-1, 0], data[n-1, 1] - yerrors[n-1] + d2]
            line_data = np.array([line_start, line_end])
            # Compute the slope, intercept, and correlation coefficient of the line
            slope, intercept, r_value, p_value, std_err  = linregress(line_data[:,0], line_data[:,1])
            # Check which points lie within the bounds of the line and count how many there are
            num_points_within_bounds = sum([is_point_in_line_bounds(data[i, 0], data[i, 1], slope, intercept, yerrors[i]) for i in range(n)])
            # If all points lie within the bounds of the line, we're done
            if num_points_within_bounds == n:
                return True, True, slope, intercept
            # If not, mark the points that are outside the bounds of the line for dropping
            else:
                for i in range(n):
                    if not is_point_in_line_bounds(data[i, 0], data[i, 1], slope, intercept, yerrors[i]):
                        n_drop[i] += 1
                    itera+=1
            d1 += 0.001
        # If we've reached the end of the line, move the line up
        if (data[n-1, 1] - yerrors[n-1] + d2) <= (data[n-1, 1] + yerrors[n-1]):
            d2 += 0.001
        # If we've reached the end of the data, drop the point with the most marks and try again
        else:
            print("The condition is fail")
            print("The following points were dropped:")
            m = np.max(n_drop)
            for i in range(n):
                if n_drop[i] == m:
                    print(f"({data[i,0]}, {data[i,1]})")
                    data = np.delete(data, i, axis=0)
                    yerrors = np.delete(yerrors, i)
                    n=len(data)
                    return data, yerrors, False, False # break out of the loop after dropping the first point with the maximum number of marks
    #return data, yerrors, slope, intercept


# If a file was uploaded
if file:
    # Read the file into a Pandas DataFrame
    df = pd.read_excel(file, engine='openpyxl')

    # Display the DataFrame in the app
    #st.write(df.to_numpy())
    df1=df.to_numpy()
    data=df1[:,0:2]
    #st.write(data)
    yerrors = df1[:,2]
    n=len(yerrors)
    slope, intercept, r_value, p_value, std_err = linregress(data[:,0], data[:,1])
    if df1[:,1].mean<0.01:
        st.write("slope for best fit : {:.3f}".format(slope))
        st.write("intercept for best fit {:.3f}: ".format(intercept))
        st.write("R square : {:.3f}".format(r_value))
    else:
        st.write("slope for best fit : {:.2f}".format(slope))
        st.write("intercept for best fit {:.2f}: ".format(intercept))
        st.write("R square : {:.2f}".format(r_value))

    #data1 = np.array([[data[0, 0], data[0, 1] + yerrors[0]], [data[n-1, 0], data[n-1, 1] - yerrors[n-1]]])
    #data2 = np.array([[data[0, 0], data[0, 1] - yerrors[0]], [data[n-1, 0], data[n-1, 1] + yerrors[n-1]]])

#slope max
    data,yerrors,slope1,intercept1=find_max_slope(data,yerrors,1000)
    while slope1==False:
      data,yerrors,slope1,intercept1=find_max_slope(data,yerrors,1000)
    if df1[:,1].mean<0.01:
        st.write("slope max : {:.3f}".format(slope1))
    else:
        st.write("slope max : {:.2f}".format(slope1))
    #x, y = data[:, 0], data[:, 1]

    # Fit line
    #reg1 = np.polyfit(x1, y1, 1)
    #fitline1 = np.polyval(reg1, x1)
    b1=intercept1
    m1=slope1


#slope min
    data=df1[:,0:2]
    yerrors = df1[:,2]
    x, y = data[:, 0], data[:, 1]
    data,yerrors,slope2,intercept2=find_min_slope(data,yerrors,1000)
    while slope2==False:
      data,yerrors,slope2,intercept2=find_min_slope(data,yerrors,1000)
    if df1[:,1].mean<0.01:
        st.write("slope min : {:.3f}".format(slope2))
    else:
        st.write("slope min : {:.2f}".format(slope2))
    #x, y = data[:, 0], data[:, 1]

    # Fit line
    reg = np.polyfit(x, y, 1)
    fitline = np.polyval(reg, x)
    b=intercept2
    m=slope2
# Plot
    yerrors = df1[:,2]
    plt.errorbar(x, y, yerr=yerrors, fmt='o', color='r')
    if df1[:,1].mean<0.01:
        plt.plot(x, fitline, label=f"Fit line: {reg[0]:.3f}x + {reg[1]:.3f}")
        plt.plot(x, b + m*x, label=f"Fit slope min: {m:.3f}x + {b:.3f}")
        plt.plot(x, b1 + m1*x, label=f"Fit slope max: {m1:.3f}x + {b1:.3f}")
    else:
        plt.plot(x, fitline, label=f"Fit line: {reg[0]:.2f}x + {reg[1]:.2f}")
        plt.plot(x, b + m*x, label=f"Fit slope min: {m:.2f}x + {b:.2f}")
        plt.plot(x, b1 + m1*x, label=f"Fit slope max: {m1:.2f}x + {b1:.2f}")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(plt.show())


