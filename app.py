import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    height = request.form.get('height')
    weight = request.form.get('weight')
    footsize = request.form.get('footsize')
    

    """#### predict gender using height, weight and foot size"""

    data = pd.DataFrame()

    # target var
    data['Gender'] = ['male','male','male','male','female','female','female','female']

    # feature vars
    data['Height'] = [6,5.92,5.58,5.92,5,5.5,5.42,5.75]
    data['Weight'] = [180,190,170,165,100,150,130,150]
    data['Foot_Size'] = [12,11,12,10,6,8,7,9]


    # Compare with person
    person = pd.DataFrame()

    person['Height'] = [float(height)]
    person['Weight'] = [float(weight)]
    person['Foot_Size'] = [float(footsize)]


    # Calculate priors
    n_male = data['Gender'][data['Gender'] == 'male'].count()
    n_female = data['Gender'][data['Gender'] == 'female'].count()
    total_ppl = data['Gender'].count()


    #Probability of male or female
    P_male = n_male/total_ppl
    P_female = n_female/total_ppl

    # Group the data by gender and calculate the means of each feature
    data_means = data.groupby('Gender').mean()


    # Group the data by gender and calculate the variance of each feature
    data_variance = data.groupby('Gender').var()


    """#### Create vars for height, weight, foot size for male and female"""

    # Means for male
    male_height_mean = data_means['Height'][data_variance.index == 'male'].values[0]
    male_weight_mean = data_means['Weight'][data_variance.index == 'male'].values[0]
    male_footsize_mean = data_means['Foot_Size'][data_variance.index == 'male'].values[0]

    # Variance for male
    male_height_variance = data_variance['Height'][data_variance.index == 'male'].values[0]
    male_weight_variance = data_variance['Weight'][data_variance.index == 'male'].values[0]
    male_footsize_variance = data_variance['Foot_Size'][data_variance.index == 'male'].values[0]

    # Means for female
    female_height_mean = data_means['Height'][data_variance.index == 'female'].values[0]
    female_weight_mean = data_means['Weight'][data_variance.index == 'female'].values[0]
    female_footsize_mean = data_means['Foot_Size'][data_variance.index == 'female'].values[0]

    # Variance for female
    female_height_variance = data_variance['Height'][data_variance.index == 'female'].values[0]
    female_weight_variance = data_variance['Weight'][data_variance.index == 'female'].values[0]
    female_footsize_variance = data_variance['Foot_Size'][data_variance.index == 'female'].values[0]

    # eg, P(Height | Female)
    # Create a function that calculates p(x | y):
    def p_x_given_y(x, mean_y, variance_y):

        # Input the arguments into a probability density function
        p = 1/(np.sqrt(2*np.pi*variance_y)) * np.exp((-(x-mean_y)**2)/(2*variance_y))
        
        # return p
        return p

    #Check if male
    # Numerator of the posterior if the unclassified observation is a male
    a=(
    P_male * \
    p_x_given_y(person['Height'][0], male_height_mean, male_height_variance) * \
    p_x_given_y(person['Weight'][0], male_weight_mean, male_weight_variance) * \
    p_x_given_y(person['Foot_Size'][0], male_footsize_mean, male_footsize_variance))

    # Check if female
    # Numerator of the posterior if the unclassified observation is a female
    b=(
    P_female * \
    p_x_given_y(person['Height'][0], female_height_mean, female_height_variance) * \
    p_x_given_y(person['Weight'][0], female_weight_mean, female_weight_variance) * \
    p_x_given_y(person['Foot_Size'][0], female_footsize_mean, female_footsize_variance))

    if (a > b):
        return render_template('index.html', prediction_text='You are Male')
    else:
        return render_template('index.html', prediction_text='You are Female')


if __name__ == "__main__":
    app.run(debug=True)