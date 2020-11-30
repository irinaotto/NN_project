#Developed by Irina Petrakova-Otto
# Inspired by https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/
# and https://wonikjang.github.io/deeplearning_unsupervised_som/2017/06/30/som.html

import matplotlib.pyplot as plt
import numpy as np
import random as ran
import tensorflow_datasets as tfds
import tensorflow as tf
import pandas as pd


class SOM(object):

    # To check if the SOM has been trained
    trained = False

    def __init__(self, m, n, dim, n_iterations=100, alpha=None, sigma=None):

        # Assign required variables first
        self.m = m; self.n = n
        self.dim=dim
        if alpha is None:
            self.alpha = 0.2
        else:
            self.alpha = float(alpha)
        if sigma is None:
            self.sigma = max(m, n) / 2.0
        else:
            self.sigma = float(sigma)
        self.n_iterations = abs(int(n_iterations))

        self.graph = tf.Graph()
        self.weightage_vects = tf.random.normal( [m * n, dim]) 
        self.location_vects = tf.constant(np.array(list(self.neuron_locations(m, n))))

       
    def calculate_BMI(self, vector_input):
        distance=[]
        for index in range(self.weightage_vects.shape[0]) :
            distance.append(tf.pow(tf.math.subtract(self.weightage_vects[index], vector_input), 2))
        bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(distance, 1)), 0) 
        slice_input = tf.pad(tf.reshape(bmu_index, [1]), np.array([[0, 1]]))
        bmu_loc = tf.reshape(tf.slice(self.location_vects, 
                                    slice_input, tf.constant(np.array([1, 2]), dtype=tf.int64)), [2])
        return bmu_index, slice_input,bmu_loc
    
    
    def calculate_learning_rate(self, iter_input, bmu_loc):
         # To compute the alpha and sigma values based on iteration number
        learning_rate_op = tf.subtract(1.0, tf.divide(iter_input, self.n_iterations))
        alpha_op = tf.multiply(self.alpha, learning_rate_op)
        sigma_op = tf.multiply(self.sigma, learning_rate_op)
        distance=[]
        # learning rates for all neurons, based on iteration number and location w.r.t. BMU.
        for index in range(self.location_vects.shape[0]):
            distance.append(tf.pow(tf.subtract(self.location_vects[index], bmu_loc ) , 2 ))

        bmu_distance_squares = tf.reduce_sum(distance, 1)
        neighbourhood_func = tf.exp(tf.negative(tf.divide(tf.cast(
                bmu_distance_squares, "float64"), tf.pow(sigma_op, 2))))
        learning_rate_op = tf.multiply(alpha_op, neighbourhood_func)

        # Finally, the op that will use learning_rate_op to update the weightage vectors of all neurons
        learning_rate_multiplier=[]
        for i in range(self.m*self.n):
            learning_rate_multiplier.append([tf.tile(tf.slice(
            learning_rate_op, np.array([i]), np.array([1])), [self.dim])] )
        return learning_rate_multiplier
   
    def calculate_delta_weights(self, learning_rate_multiplier, vect_input):
         # W_delta = L(t) * ( V(t)-W(t) )
        difference=[]
        weight_delta=[]
        for index in range(self.weightage_vects.shape[0]) :
            difference=tf.subtract(vect_input, self.weightage_vects[index])
            learning_rate=tf.multiply(learning_rate_multiplier[index], difference)
            weight_delta.append((tf.transpose(learning_rate)).numpy().flatten())
        return weight_delta 
    def update_weights (self, weight_delta):
             # W(t+1) = W(t) + W_delta
        self.weightage_vects = tf.add(self.weightage_vects,tf.cast(weight_delta, float))
        
   
    def neuron_locations(self, m, n):

        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

    def train(self, input_vects):

        for iter_no in range(self.n_iterations):
            # Train with each vector one by one
            for input_vect in input_vects:
                #self.sess.run(self.training_op, 
                #       feed_dict={self.vect_input: input_vect, self.iter_input: iter_no})
                input_array=(tfds.as_numpy(input_vect['image'])).flatten()
                bmi_index, slice_input, bmu_loc = self.calculate_BMI(input_array)
                learning_rate_multiplier=self.calculate_learning_rate(iter_no, bmu_loc)
                new_weight_op=self.calculate_delta_weights(learning_rate_multiplier, input_array )
                self.update_weights(new_weight_op)

        # Store a centroid grid for easy retrieval later on
        centroid_grid = [[] for i in range(self.m)]
        self.weightages = list(self.weightage_vects)
        self.locations = list(self.location_vects)
        for i, loc in enumerate(self.locations):
            centroid_grid[loc[0]].append(self.weightages[i])

        self.centroid_grid = centroid_grid

        self.trained = True

    def get_centroids(self):

        if not self.trained:
            raise ValueError("SOM not trained yet")
        return self.centroid_grid

    def map_vects(self, input_vects):

        if not self.trained:
            raise ValueError("SOM not trained yet")

        to_return = []
        labels=[]
        for vect in input_vects:
            images=(tfds.as_numpy(vect['image'])).flatten()
            min_index = min( [i for i in range(len(self.weightages))], 
                            key=lambda x: np.linalg.norm(images - self.weightages[x]) )
            to_return.append(self.locations[min_index])
            labels.append(vect['label'].numpy())

        return to_return, labels
    def set_labels(self, centroids, labels):
        self.centroid_label=pd.DataFrame({'x':centroids[:,0],'y':centroids[:,1],'label': labels})
    def set_centroids_with_label_prob(self):
        self.centroids_with_label_prob=pd.DataFrame(columns=['x','y','label','prob'])
        #for each neuron calculate most probable value and store probability and most possible value
        for x in range(self.m):
            for y in range(self.n):
                centroid=self.centroid_label[(self.centroid_label['x']==x) & (self.centroid_label['y']==y)]
                if len(centroid)>0:
                    labels=centroid['label']
                    labels=pd.DataFrame(labels, columns=['label'])
                    pivot_table=labels.pivot_table(index=['label'], aggfunc='size')
                    most_likely_label=int(pivot_table.idxmax())
                    prob=pivot_table[pivot_table.idxmax()]/pivot_table.sum()
                    self.centroids_with_label_prob=self.centroids_with_label_prob.append({'x':x,'y':y,'label':most_likely_label,'prob':prob}, ignore_index=True)
        return self.centroids_with_label_prob
    
    def get_centroids_with_label_prob(self):
        return self.centroids_with_label_prob          
    def get_centroids_labels(self):
        return self.centroid_label
    def predict(self, centroid):
        most_likely_label=-1
        prob=0
        x=self.centroids_with_label_prob[(self.centroids_with_label_prob['x']==centroid[0]) & (self.centroids_with_label_prob['y']==centroid[1])]
        if len(x)>0:
            most_likely_label=int((x['label'].to_numpy())[0])
            prob=(x['prob'].to_numpy())[0]
        return most_likely_label,prob
'''
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=False,
    with_info=True,
)

training_data=ds_train.take(100)
my_som = SOM(10, 10, 28*28, 5)
my_som.train(training_data)
# Fit train data into SOM lattice

mapped, labels = my_som.map_vects(training_data)
mappedarr = np.array(mapped)
my_som.set_labels(mappedarr, labels)
my_som.set_centroids_with_label_prob()

#PREDICT ON TEST DATA
test_data=ds_test.take(5)
test_mapped, test_labels=my_som.map_vects(test_data)
test=np.array(test_mapped)
accuracy=0
for index in range(len(test)):
    predicted_label,prob=my_som.predict(test[index])
    if predicted_label==test_labels[index]:
        accuracy=accuracy+1
centroids_with_labels=my_som.get_centroids_with_label_prob()

## Plots: 1) Train 2) Test+Train ###

x1 = mappedarr[:,0]; y1 = mappedarr[:,1]
plt.figure(1, figsize=(12,6))
plt.subplot(121)
# Plot 1 for Training only
plt.scatter(x1,y1)
# Just adding text
for i, m in enumerate(mappedarr):
   plt.text( m[0], m[1],labels[i].numpy(), ha='center', va='center', bbox=dict(facecolor='green', alpha=0.5, lw=0))
plt.title('Train MNIST 100')
plt.show()

x1=centroids_with_labels['x']
y1=centroids_with_labels['y']
plt.figure(1, figsize=(20,20))
plt.subplot(121)
# Plot 1 for Training only
plt.scatter(x1,y1)
# Just adding text
col='green'
for i in centroids_with_labels.index:
    if centroids_with_labels['prob'][i]==1:
        col='green'
    elif (centroids_with_labels['prob'][i]<1 and centroids_with_labels['prob'][i]>0.5):
        col='blue'
    else:
        col='red'
    plt.text(centroids_with_labels['x'][i],centroids_with_labels['y'][i],int(centroids_with_labels['label'][i]),
                ha='center', va='center', bbox=dict(facecolor=col, alpha=1, lw=0))
plt.title('Train MNIST 100')
plt.show()
'''
