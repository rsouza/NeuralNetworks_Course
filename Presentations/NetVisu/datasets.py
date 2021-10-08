import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

def entangled_two_circles(number_of_points_per_class, plot = True):
    '''
    Creates the dataset of two circles in R^3, a blue and a red one, each inside of the other.
    Inputs: number of points in each class of the dataset, if plot is true, shows the dataset in R^3
    Outputs: (dataset in 3 coordinates, labels of the dataset)
    '''
    theta = np.linspace(0,2*np.pi,number_of_points_per_class)
    x_1 = np.cos(theta)
    y_1 = np.sin(theta)
    z_1 = np.zeros(len(theta))

    x_2 = np.zeros(len(theta))
    y_2 = np.cos(theta)+np.ones(len(theta))
    z_2 = np.sin(theta)

    x = np.concatenate((x_1, x_2))
    y = np.concatenate((y_1, y_2))
    z = np.concatenate((z_1, z_2))

    target = np.concatenate((np.zeros(len(x_1)), np.ones(len(x_2))))

    x = np.reshape(x,(-1,1))
    y = np.reshape(y,(-1,1))
    z = np.reshape(z,(-1,1))
    data = np.concatenate((x,y,z), axis = 1)

    if plot == True:
        fig = plt.figure()
        ax = Axes3D(fig)

        scatter = ax.scatter(x_1,y_1,z_1, c= 'r')
        scatter = ax.scatter(x_2,y_2,z_2, c= 'b')
        plt.show()
    
    return (data, target)

def entangled_four_circles(number_of_points_per_class, plot = True):
    '''
    Creates the dataset of four circles in R^3, two blue and two larger red one, each entangled inside the other.
    Inputs: 4*(number of points) in each class of the dataset, if plot is true, shows the dataset in R^3
    Outputs: (dataset in 3 coordinates, labels of the dataset)
    '''
    theta = np.linspace(0,2*np.pi,number_of_points_per_class)
    
    x_r1 = 1/2*np.cos(theta)
    y_r1 = 1/2*np.sin(theta)+np.ones(len(theta))
    z_r1 = np.zeros(len(theta))

    x_r2 = 1/2*np.cos(theta)
    y_r2 = 1/2*np.sin(theta)-np.ones(len(theta))
    z_r2 = np.zeros(len(theta))

    x_r = np.concatenate((x_r1, x_r2))
    y_r = np.concatenate((y_r1, y_r2))
    z_r = np.concatenate((z_r1, z_r2))

    x_b1 = 1/3*np.ones(len(theta))
    y_b1 = np.cos(theta)
    z_b1 = np.sin(theta)

    x_b2 = -1/3*np.ones(len(theta))
    y_b2 = np.cos(theta)
    z_b2 = np.sin(theta)

    x_b = np.concatenate((x_b1, x_b2))
    y_b = np.concatenate((y_b1, y_b2))
    z_b = np.concatenate((z_b1, z_b2))
    
    x = np.concatenate((x_r, x_b))
    y = np.concatenate((y_r, y_b))
    z = np.concatenate((z_r, z_b))
    
    target = np.concatenate((np.zeros(len(x_r)), np.ones(len(x_b))))

    x = np.reshape(x,(-1,1))
    y = np.reshape(y,(-1,1))
    z = np.reshape(z,(-1,1))
    data = np.concatenate((x,y,z), axis = 1)
    
    if plot == True:
        fig = plt.figure()
        ax = Axes3D(fig)

        scatter = ax.scatter(x_r,y_r,z_r, c= 'r',  alpha = 0.5)
        scatter = ax.scatter(x_b,y_b,z_b, c= 'b')
        
    return (data, target)

def circle_inside_torus(number_of_points_per_class, plot = True):
    '''
    Creates a dataset of a circle of radius 2 inside a torus of inner radius 1 and outter radius 3 of number_of_points_per_class
    in each of these manifolds. If plot = True, it plots both the surface and the scatter plot for the torus sampled data
    Inputs: number_of_points_per_class; plot = True
    '''
    theta_circle = np.linspace(0,2*np.pi,number_of_points_per_class) 
    x_circle = 2*np.cos(theta_circle)
    y_circle = 2*np.sin(theta_circle)
    z_circle = np.zeros(len(theta_circle))
    x_circle = np.reshape(x_circle,(-1,1))
    y_circle = np.reshape(y_circle,(-1,1))
    z_circle = np.reshape(z_circle,(-1,1))
    data_circle = np.concatenate((x_circle,y_circle,z_circle), axis = 1)
    
    theta_torus = 2*np.pi*np.random.rand(round(number_of_points_per_class**1/2)) #takes the square root so that each class has the same
    #number of instances
    phi_torus = 2*np.pi*np.random.rand(round(number_of_points_per_class**1/2))
    c, a = 2, 1
    x_torus = (c + a*np.cos(theta_torus)) * np.cos(phi_torus)
    y_torus = (c + a*np.cos(theta_torus)) * np.sin(phi_torus)
    z_torus = a * np.sin(theta_torus)
    x_torus = np.reshape(x_torus,(-1,1))
    y_torus = np.reshape(y_torus,(-1,1))
    z_torus = np.reshape(z_torus,(-1,1))
    data_torus = np.concatenate((x_torus,y_torus,z_torus), axis = 1)
    
    
    data =  np.concatenate((data_circle, data_torus), axis = 0)
    target = np.concatenate((np.zeros(number_of_points_per_class),np.ones(len(data_torus))), axis = 0)

    if plot == True:
        n = 100 #we will make 100 points so to make the surface plot visible
        theta_plot = np.linspace(0,2*np.pi,n)
        phi_plot = np.linspace(0, 2*np.pi, n)
        theta_plot, phi_plot = np.meshgrid(theta_plot, phi_plot)
        c, a = 2, 1
        x_torus_plot = (c + a*np.cos(theta_plot)) * np.cos(phi_plot)
        y_torus_plot = (c + a*np.cos(theta_plot)) * np.sin(phi_plot)
        z_torus_plot = a * np.sin(theta_plot)
        
        fig = plt.figure(figsize = [10,10])
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.scatter(x_circle,y_circle,z_circle, c= 'r',alpha=1)
        ax.plot_surface(x_torus_plot, y_torus_plot, z_torus_plot,
                                  color='b', alpha =0.4, rstride=5, cstride=5, edgecolors='w')
        ax.set_zlim(-3,3)

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.scatter(x_circle,y_circle,z_circle, c= 'r',alpha=1)
        ax.scatter(x_torus, y_torus, z_torus,
                                  color='b', alpha =0.4)
        ax.set_zlim(-3,3)

    return (data, target)

def swiss_roll(number_of_points_per_class, plot = True):
    '''
    Creates a dataset of a swiss roll of number_of_class points of half of its is in class red and half in class blue. 
    If plot = True, it plots both the scatter plot for the sampled data
    Inputs: number_of_points_per_class; plot = True
    '''
    #makes the cross section of the swiss roll (ie. spiral)
    theta_blue = np.linspace(0, 5*np.pi, round(number_of_points_per_class/10))
    theta_red = np.linspace(5*np.pi, 10*np.pi, round(number_of_points_per_class/10))
    
    x_blue =  theta_blue*np.cos(theta_blue)
    y_blue =  theta_blue*np.sin(theta_blue)
    z_blue = np.zeros(len(x_blue))

    x_red =  theta_red*np.cos(theta_red)
    y_red =  theta_red*np.sin(theta_red)
    z_red = np.zeros(len(x_red))
    
    #reshape arrays
    x_blue = np.reshape(x_blue,(-1,1))
    y_blue = np.reshape(y_blue,(-1,1))
    z_blue = np.reshape(z_blue,(-1,1))
    
    x_red = np.reshape(x_red,(-1,1))
    y_red = np.reshape(y_red,(-1,1))
    z_red = np.reshape(z_red,(-1,1))

    
    blue_coord = np.concatenate((x_blue, y_blue, z_blue), axis = 1)
    red_coord = np.concatenate((x_red, y_red, z_red), axis = 1)
    
    #lifts the spiral to the swiss roll by making 9 extra layers of points
    for i in range(1, 10):
        z_new = i*np.ones(len(x_blue))
        z_new = np.reshape(z_new, (-1,1))
        
        new_blue_cord = np.concatenate((x_blue, y_blue, z_new), axis = 1)
        new_red_cord = np.concatenate((x_red, y_red, z_new), axis = 1)
        
        blue_coord = np.concatenate((blue_coord, new_blue_cord), axis = 0)
        red_coord = np.concatenate((red_coord, new_red_cord), axis = 0)
      
    #makes dataset and target
    data = np.concatenate((red_coord, blue_coord), axis = 0)
    
    target_red = np.zeros(len(red_coord))
    target_blue = np.ones(len(blue_coord))
    target = np.concatenate((target_red, target_blue), axis = 0)
    if plot == True:
        fig = plt.figure()
        ax = Axes3D(fig)

        scatter = ax.scatter(blue_coord[:,0],blue_coord[:,1],blue_coord[:,2], c = 'b')
        scatter = ax.scatter(red_coord[:,0],red_coord[:,1],red_coord[:,2], c = 'r')
        plt.show()
    
    
    return (data, target)