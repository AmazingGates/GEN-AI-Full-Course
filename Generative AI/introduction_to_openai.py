# This section will be an Introduction to OpenAI and OpenAI API's.

# First we will have a recap of everything that we have gone over in the previous section, to make sure that we have 
#a full understanding of the concepts that we were introduced to.

# Now that our recap is done, we can start moving into the practical aspect of Generative AI.

# These are the topics that we will be gong over in this section.

#   Topics -

# - OpenAI - We will have a complete walthrough of the OpenAI website.

# - OpenAI API - We will go over the steps on how we can use the OpenAI API. We will using this API with Python

# - OpenAI Playground - Here, we will go over the steps on how we can use different models, how we can pass different
#prompts, how we can generate outputs, how we can set-up different sentimients, etc etc.

# - Chat Completion API - We will go over the steps on how we can use this to call the GPT model, like OpenAI API, 
#for example, but we can use any sort of GPT model.

# - Function Call 

# This is the agenda for this section.

# Before starting with the OpenAI, we will discuss its importance.

# Alongside OpenAI, we will be dicussing topics like Hugging Face.

# Huggy Face provides a hub for all the models.

# Here we will find all the open source models.

# So we will be generating a hugging face key, and we can utilize al sorts of models that the hugging face provides.

# This is the site we can reach the hugging face enviornment. Huggingface.co/models

# This will redirect us to the particular model hub.

# We will be going over the steps of using the models on this site for different task.

# We will not be restricted to the models only on OpenAI.

# We will also explore some other open source models.

# This is another site that we will be exploring AI21 Studio.

# This site will give us access to other Large Languge Models for particular task.


# So, what is the differnce between Hugging Face and OpenAI?

# The Hugging Face models are free.

# Hugging Face is a similar enviorment to Github. We have to generate an API key to personalize our hub and use the
#models and different features associated with the Hugging Face enviornment.

# Also, by generating the OpenAI API key, we will have access to the different models and features associated with
#the OpenAI hub.

# OpenAI has many various models for many different task. 

# We can reach the OpenAI enviornment with this link OpenAI.com

# The major difference between the two is the OpenAI only has access to OpenAI models, while Hugging Face has access
#to many the models, this includes open source models, and even models from OpenAI.


# Here is a brief introduction of the OpenAI, and why OpenAI is so important.

#   About -

# - OpenAI is a leading company in the field of AI. It was founded in 2015 as a non profit organization by
#Sam Altman and Elon Musk.

# As of 2021, the Ceo of OpenAi has been Sam Altman.

# The company was founded with the goal of developing and promoting friendly AI in a responsible way, with a 
#focus on transparency and open research.
 
#   OpenAI Milestones -

# - Generative Models June 16, 2016
# - Ingredients for Robotics Research February 26, 2018
# - Solving Rubiks Cube With A Robotic Hand October 15, 2019
# - Multimodal Neurons in Artificial Neural Networks March 4, 2021


# In 2020, OpenAI launched its Chat-GPT.

# This was a major breakthrough and milestone in the history of AI.

# Now we will go over the process of generating the OpenAI API keys and utilizing them.


# Let's look at the Quickstart inside the OpenAI website.

# If we want to explore this OpenAI API, we just click on Quickstart.

# And after clicking on the Quickstart we will get all the code we need to run the model inside our sysytem.

# So if we want to use this OpenAI API, and we want to run the code, for that we just need to click on the 
#Quickstart and follow the instructions to implement the code in our program to run it.

# We will be selecting Python.

# These steps will go over the process of requesting the OpenAI API for different models.

# Next we will be exploring models, which will give us a complete breakdown of all the models we may want to use.

# Now let's begin the practical implementation by listing the steps we need to take.

# 1. Anaconda - This is Data package for the Data Science that we need in our system

# 2. Python - Must be installed 

# 3. Jupyter Notebook - This is where we will perform our practicals
# 3.(A) We will be doing our projects in VS Code

# 4. Conda - This is how we will create our virtual enviornment
# 4.(A) Then we need to activate the virtual enviornment and install all the packages we will need.

# After these steps are done we will begin with the practical implementation.

# First we will search our system for Anaconda Prompt.

# This is where we will create our virtual enviornment.

# Our first command will be conda create -n, followed directly by our enviornments name.

# conda create -n testingopenai followed by our version of Python we are using.

# The best version to use for Python in this enviornment are 3.7, 3.8, 3.9, 3.10

# conda create -n testingopenai python=3.8

# Now we will hit enter, and if everything goes correctly, our enviornment will begin to be created.

# Next we will activate the enviornment by running the activate command.

# The name of our testing enviornment should now be listed where our (base) was.

# Next we can clear the screen using the cls command.

# Note: I'll wait to do this live so that we can see the steps it took to get here.

# Next we will check the libraries that we have in enviornment by typing the command pip list

# This will give us a list of the libraries we have as of now.

# Our next step will be to download the Jupyter Notebook

# We will use this command to install it. pip install jupyter notebook

# Now we will search pip list again to see what libraries we have access to initially.

# Now we can open the notebook by entering the command jupyter notebook

# This will open our Jupyter Notebook in a new tab in our browswer

# After that we need to lauch our file by clicking the "New" dropdown tab and selecting notebook.

# This is where we will be performing all of our practicals.

# Now we will name our enviornment to reflect our training enviornment.

# Now our Notebook is ready to go.

# Now we can head back over to the OpenAI website and click on the Quickstart option and follow the instructions
#for the Python selection.

# Note: Since we are using Anaconda and already set up our own enviornment, we can skip right to step for installing
#openai.

# To install we will have to open a new anaconda prompt since we can't stop the first one because it's running our
#jupyter notebook.

# Once our new prompt is open we will type the command conda env list, to view a list of our enviornments.

# Now we select our enviornment by typing the command conda activate, followed by our enviornments name.

# conda activate testingopenai

# Now we can run the pip install command for the openai

# Now to check that we successfully installed the package, we will go back over to our Jupyter Notebook
#and import the openai library.


# So let's start with the Openai API.

# 1. What is OpenAI API

# This OpenAI API has been designed to provide developers with seamless access to state of art pre-trained, 
#artificial intelligence models like GPT-3, GPT-4, Dall-E, Whisper, Embeddings, etc. So by using this openai api we 
#can integrate cutting edge ai capabilities into our applications regardless of the program language.

# So, the conclusion is by using this OpenAI API we can unlock the advanced functionalities and we can enhance the 
#intelligence and performance of our applications.

# Next we will go over generating an OpenAI API Key.

# Without this we cannot use the OpenAI API.

# So here is the process to generate our Key.

# First we go back to the OpenAI website.

# Then we will find the API key option and click on it. This should be located under the dashboard.

# Now follow the instructions to generate your new private keys and store them in a secure location.

# Next we will paste our API key in jupyter notebook.

# Now what we need to do after generating our API key and bringing it over to our jupyter notebook is call the 
#OpenAI API.

# This is how we do that using these lines of code that we will write in jupyter notebook.

# openai.api_key = (The name of your API key)

# Next we will run openai.models.list().

# This will display all the models we have access to.

# We can go a step further and convert this data into a more readable list.

# We'll use these steps to do that.

# We'll create a variable for openai.models.list() and call it all_models.

# all_models = openai.models.list()

# Now we will create a list method and pass in our variable (all_models).

# list(all_models)

# This gives us all the models in a more readable format.

# From here we can create a data frame also.

# To do that first we'll start by importing pandas as pd

# Then we will enter this method pd.DataFrame(), which we will pass list(all_models)

# This will give us our data in a formatted frame.

# We can also ad the column names manually.

# We will add columns=["id", "created", "object", "owned_by"] to the end of pd.DataFrame(), inside the last 
#parenthesis.

# Next we will explore the openai playground.

#       OpenAI Playground

# How to open the openai playground: https://platform.openai.com/playground?model=assistant

# In chat there is an option of system. This pertains to how we want our chatbot to behave.

# Once we navigate to the playground, we will select the chat button, and in the system box, we describe how we 
#want our chatbot to behave.

# This is one of three fields in our playground.

# The other two are Chat, and User.

# We will go over all of the fields.

# So for system, since this sets the behavior for chatbot, we will write, "You are a helpful assistant".

# Now we go the user button. This button will toggle between user and assistant once clicked.

# Since we will be asking questions we will leave it user.

# Now we will type in a simple question.

# So this is the question we are asking our chat-gpt

# Next we can select chat to pick the model we want to use.

# We will be using the chat-gpt 3.5 turbo model.

# We also have various other options pertaining to the model selection. One of them is temperature.

# The temperature controls the randomness. This controls whether the chatbot will be creative in its responses,
#or give staright forward answers.

# That is one example. You can hover over each for field for a description.

# Once we run the code we can actually use the viewcode button to see the entire python code of the process we 
#just ran.

# Now, there other topics that we will be going over once we get to the section where we are building our project.

# But for now, let's keep this one term in mind, R A G.

# R A G - Retrieval-Augmented Generation: Is an artificial intelligence framework that retrieves data from external
#sources of knowledge to improve the quality of responses. This natural language processing technique is commonly
#used to make large language models more accurate and up to date.


# Next we will delete everything from the pyton code we copied over and keep only the role: user and the content:, 
#which is our input prompt.

# Note: This prompt is pretty simple and later on we will go over the process create different types of prompts.

# Also note that we can't access the gpt from openai website at the moment without having a payment method
#selected.

# But will continue to take notes and and go over the steps for when we update our infrmation or use an
#open source model which doesn't require a payment method.

# After running our prompt, we should be given a respone from the calling the gpt from the openai.

# Note: We can add type(response), and this should give us the type of response we are given.

# Next we can extract the response from all of the information we are given back in return by using
#response.choices[0] to extract the first index of the choice list, and then adding .message to call the message
#which contains the information we are looking for.

# response.choices[0].message

# This will return our generated return message for the prompt we entered.

# Also, we can add a .content at the end of that to get back just the returned message for our prompt.

# response.choices[0].meassge.content

# Note: We can aslo describe the parameters that we want by specifying it the code where our prompt is.

# Defining these parameters will alter our output prompts.

# These are all the parameters we can use.

# model = "yes"
# prompt = input prompt
# max_tokens = in how many number of tokens we want result
# temperature = for getting some creative output
# n = number of outputs

# So for now we'll just define the max_tokens and the n.

# We will do this in the same box as our prompt code.

# See box 24 in our jupyter notebook.

# Note: There is a Tokenizer built into the OpenAI website that allows us to see how many tokens we will be charged
#per input prompt and output prompt.


# In this section we will be going over Function Calling.

# This is a very important feature of the openai api.

# If we're going to use the openai api then we definitely need to know about function calling.

# By using the function calling we can do multiple things.

# Here we will go over all the things we can do with the function calling.

# The second thing we will be going over here is Lang Chain.

# In Lang Chain we will be discussing how we can use the openai via Lang Chain.

# We will discuss the differnces between Lang Chain and the OpenAI and why we should use Lang Chain and what the 
#benefits of using Lang Chain and things we can do.

# We will also go over the steps of creating an end to end application using Lang Chain.

# Lang Chain is going to be very very important part if we are going to learn Generative AI, if we are talking 
#about LLM's, and if we are going to build any sort of application.

# So along with the openai, lang chain plays a very important role.

# After That we will be going over the concept of Prompt Templates.

# This will discuss how we can design different kinds of prompts.

# This is when we will going the use of Hugging Face.

# We will going over the steps of how we can utilize Hugging Face models.

# This will teach us how to use any open source model.

# We will do this by generating Hugging Face API keys, and by utilizing that API key we can access any type of model
#that is available on Hugging Face.

# So Basically we will be using Hugging Face with Lang Chain.

# And Then we will further discuss a few more Lang Chain concepts that will be very important. We will list those 
#topics below.

# 1. Chain

# 2. Agent - How we can create agents, and how we can use agents.

# 3. Memory - We will learn how to create memory using Lang Chain.

# These are all very important parts of Lang Chain.

# Without knowing these steps we can not develope any sort of applications.

# So we will discuss all of these topics in full before we start our end to end project.

# Now we will go over to jupyter notebook and change our notebooks name to add Lang Chain since we will be discussing
#this from here on out.

# Next we will import lang chain into our notebook.

# Now that we have our Lang Chain import we can dive into Function Calling.

# To get a detailed overview of Function Calling we can go to the OpenAI website and locate it under the DOC's
#section.

# Here are a few bullentin points we can take away from the overview.

# 1. The basic use of the function calling is that we can format our output in our desired format. So whatever output 
#we are getting from the OpenAI, for example, we can format that particular output in our required format.

# 2. Let's say that we are calling any sort of API, which means we are asking something of the chatgpt and it is
#not able to answer for that particular question. For this, what we are doing is we are calling any sort of plug in,
#and whatever output we get, we can format that output and append that output in our conversation chain.

# Now we write some code to help us better understand the function calling.

# This is the simple code we wrote in our notebook

# student_description = "Python Papi is a student of computer science at Harvard. He is American and his GPA is a 4.0.
#He has been studying computer science for about 5 years and he is in love with the field.

# Then we called that code using the student_description to get the description in a different cell in our notebook.

# This was a short description.

# Now we will define a prompt.

# And we will pass this prompt to our chat-gpt model.

# So whenever we are talking about a prompt we talking about something like this.

#   Input --------> [LLM] ---------> Output

# This is us putting in an input and generating an output.

# The input is called an input prompt, and the output is called an output prompt.

# So now we will add a simple prompt to our notebook that we will pass to our gpt model.

# This is the prompt we will use.

# "A simple prompt to extract information from 'student_description' in a JSON format"

# prompt = f'''
# please extract the following information from the given text and return it as a JSON odject:

#name
#college
#grades
#club

# This is the body of text to extract the information from:
#{student_description}
#'''

# Now that we have this prompt, we can pass it to our gpt model and it will perform this task for us.

# Now we can call our prompt in our notebook to see if everything is working fine before we pass it to our
#gpt model.

# These are the steps we will take to pass this prompt to our gpt.

# Then we will create an object for this particular class.

# Now we can keep our object inside of a variable.

# For example purposes, we will say that our variable name is client

# Then we need to connect this prompt to the openai by passing in our API key.

# Now we can run this by calling the client

# from OpenAI import OpenAI
# client = OpenAI(api_key = Keys)
# client

# Now that we are connected to our gpt model, let's write some code to call the chat-completion model.

# This is the code that we will write in our notebook.

# response = client.chat.completions.create(
#model = "gpt-3.5-turbo",
#messages = [
#{
#"role": "user",
#"content": "prompt"
#}
#]
#)

# This is how we are going to call the chat completion API.

# Note: Even though we have rub everything correct, we still haven't added a payment method to our OpenAI 
#website so it will throw an error.

# I just went through the process of everything to make sure the steps were correct for when we started using
#open source models.

# Next, we can clean up the response.

# This is how we can do that.

# response.choices 

# From here we can index into the list of choices we are returned using this method.

# response.choices[0]

# From here we extract the message from the returned information using this method.

# response.choices[0].message

# Lastly we can pull the content we want from the message using this method.

# response.choices[0].message.content

# If we wanted to convert this particular information into a JSON file, first we have to start by storing our method
#in a variable.

# output = response.choices[0].message.content

# Next we will import json

# In the same block we will write this code under our import.

# json.loads()

# Inside this loads() function we will pass our variable, output.

# This method of steps will return our content in JSON form.

# We have successfully pass our propmpt to our gpt model and have gotten back a response from a request.

# This type of prompt is called a Few Short Prompt.

# This is where we are giving a description and telling the model that this is how we want it to behave, and based 
#off of that we are expecting some information.

# On the other hand, when we ask a question directly to our model, instead of passing a prompt, this is called a
#(Zero Short Prompt).


# What is the use of the function calling?

# To see examples of function calling we will define a simple function, and then define a complex function to get
#a better visualization.

# So what we're going to do is define a function that will perform the same task as the chat.completion.create()
#we just used.

# This is the function we will use

# student_custom_function = [
#{
#   "name": "extract_student_info",
#   "description": "Get the student information from the body of the input text",
#   "parameters": {
#       "type": "object",
#       "properties": {
#           "name": {
#               "type": "string",
#               "description": "Name of the person"
#           },
#           "school": {
#               "type": "string",
#               "description": "College name"
#           },
#           "grades": {
#               "type": "Integer",
#               "description": "CPGA of the student"
#           },
#           "club": {
#               "type": "string",
#               "description": "college club for extracurricular activities"
#           },
#}
#}
#}
#]

# Now we would run this code in our notebook.

# And after we call it we will need to call the chat.completion API.

# We can just copy and paste the original response method to do this.

# Now we need to define fome sort of parameter.

# First we will condense our message into a single line.

# After the message we can write out our parameters.

# The parameters name will be entered as functions.

# We will enter them with a comma after the message.

# And the name of our function will be the same as the function we created.

# functions = student_custom_function

# Next we can rename this updated response as response2. 

# Note: Even though we got back an error everything is running fine. The error is for the subscription update
#with openai website.

# Now we can call response 2 and have our information returned.

# We will get output in whatever format we have defined our original function.

# This is a very basic use of our function.

# We will see that there are differences between the direct call and our function call.

# We will see differences when we try to get the content from response2.

# Lets look at what we get back once dig all the way down to the .content with our response2.

# We see that we don't get any information with response2.choices[0].message.content.

# This is because we were calling the API directly and so we were able to get the content from the message.

# So this (response2.choices[0].message) will still work, but since we used a function call to get information
#from the API, we will need to get the information from the function.

# That will look like this.

# response2.choices[0].message.function_call

# This will get us the output we want.

# Now we can add .arguments after that will get us our content.

# Now just like before we can add json.loads(), and pass in the response2.choices[0].message.function_call as 
#parameter to get back our content in JSON format.

# json.loads(response2.choices[0].message.function_call)


# Now we will go over another example of function calls.

# The second part of our code will be written as a for loop.

# This is our code that we will write in our notebook

# student_info = [student_one, student_two]
# for stu in student_info:
#     response = client.chat.completions.create(
#         model = "gpt-3.5-turbo1126",
#         messages = {{"role":"user", "content":"stu"}},
#         functions = function_one,
#         function_call = "auto"
# )

#    response = json.loads(response.choices[0].message.function_call.arguments)
# print(response)#import csv

# Inside this for loop we have defined a list[].

# Inside this list we have two descriptions, [student_one, student_two]

# Now we will copy and paste the student_description from our previous example.

# We will paste it above our new advanced code in our jupyter notebook.

# When we run it we should get back the original student_description that we wrote for our student_description.

# We will use this student description to replace the student_one in our student_info list.

# Now we will create one more variable and call it student_description_two, and set it equal to the 
#description from our original student_description.

# student_description_two = "Brian Javelle Gates is a student at Harvad pursuing his Maters in Business. He is a Native 
#to America and his GPA is a 4.0. He has been studying Business since Highschool and has already founded and sold many
#Companies as a young Man. He is also apart of the chess club. His goal is to dominate the Business world in all
#aspects possible and live and love life in the process with his Wife and Children.

# Now we will change up the information that we just assigned to student two so that it is distinct from the 
#information assigned to student one.

# Now we will replace the student_two in student_info list with this new student_description_two in our jupyter
#notebook.

# Now what we are going to do is change the function_one that our functions is assinged to in our basic to
#student_custom_function we created previously.

# The function_call will stay assigned to "auto".

# Now we can run the code and get back our output response.

# We should get back two respones, one for each student in our student_info list.

# Now we will add a third student to our student_info list.

# Now that we have our student_description_three defined and we have passed into our student_info list, we will
#be able to generate an output for this student as well when we run our code.

# So we will be getting back 3 output responses now.

# Now we will go over the process of calling multiple functions at once.

# Let's start by defining one more function in our notebook.

# We will use the same function that we used for our first function.

# Next we will use the same student_info list[] we created by copying and pasteing it to under our new function
#in the notebook.

# This is the code we will be copying and pasteing.

#student_info = [student_description, student_description_two, student_description_three]
#for stu in student_info:
#    response = client.chat.completions.create(
#        model = "gpt-3.5-turbo1126",
#        messages = {{"role":"user", "content":"stu"}},
#        functions = student_custom_function,
#        function_call = "auto"
#)

#    response = json.loads(response.choices[0].message.function_call.arguments)
#print(response)#import csv

# Now we can create a new list on top of the code we just pasted, but in the same block.

# And inside this list what we can do is right another function.

# This is what we'll have.

# functions = [funtion_one[0], function_two[0]]

# This is a list of our first function and our second function.

# We will pass in the first function we created (student_custom_function) and copy over the function_one[0] in 
#our new functions list.

# functions = [student_custom_function[0], function_two[0]]

# function_two[0] will stay the same because that is what we nemed our second function.

# Using this process we can call our multiple functions.

# We just need to modify one thing.

# Instead of functions = student_custom_function like we see below

#student_info = [student_description, student_description_two, student_description_three]
#for stu in student_info:
#    response = client.chat.completions.create(
#        model = "gpt-3.5-turbo1126",
#        messages = {{"role":"user", "content":"stu"}},
#        functions = student_custom_function,
#        function_call = "auto"
#)

# We will assign functions to functions because this is where we defined our new function_two list.

#student_info = [student_description, student_description_two, student_description_three]
#for stu in student_info:
#    response = client.chat.completions.create(
#        model = "gpt-3.5-turbo1126",
#        messages = {{"role":"user", "content":"stu"}},
#        functions = functions,
#        function_call = "auto"
#)

# Now we are done and we will be able to call both our functions when we run our code.



# So what is the actual use of calling functions?

# We will discuss this by going over the advanced example now.

# Fisrt we will write the code to call our gpt.

# This is the code we will use.

# response = client.chat.completions.create(
#    model = "gpt-3.5-turbo-1106",
#    messages = [
#    {
#        "role": "user",
#        "content": "When's the next flight from Amsterdam to New York?"
#    }
#    ]
#)

# This is the question we are asking our LLM model, and the question we want a response to.

# But, our model will not be able to answer this question for us. Why not?

# If we go back to the openai website and check this particular model's information we will see that this model
#is only trained on data up until the date September 2021.

# This means that we will not be able to get accurate up to date flight information using this method.

# So our goal will be to make our chat capable of handling such requests.

# This is where we discover the usefulness of function calling.

# First we will define one function.

# functions_description = [
#     {
#           "name": "get_flight_info",
#           "description": "Get flight information between two locations",
#           "parameters": {
#               "type": "object",
#               "properties": {
#                    "loc_origin": {
#                        "type": "string",
#                        "description": "The departure airport, e.g. AMS",
#                     },
#                     "loc_destination": {
#                         "type": "string",
#                         "description": "The destination airport, e.g. JFK",
#                     },
#               },
#               "required": ["loc_origin", "loc_destination"],
#           },
#     },
#]


# Now that we have our function defined we will define our prompt.

# This is the prompt we will define.

# user_prompt = "When is the next flight from Amsterdam to NYC"

# After our prompt we will copy and paste the respone again.

# response = client.chat.completions.create(
#    model = "gpt-3.5-turbo-1106",
#    messages = [
#    {
#        "role": "user",
#        "content": "When's the next flight from Amsterdam to New York?"
#    }
#    ]
#)

# Here we will change the content from "When's the next flight from Amsterdam to New York?", to user_prompt.

# Next we will add  funtions = functions_description, and function_call = "auto" inside of our respone inside
#our notebook.

# See line 461 to see our function_description

# This is our prompt.

# If we were to run it now, we would get back something different than the origingal message informing us that
#the model was unable to provide the data we requested because it was not trained of present day data.

# First we will write response2.choices[0].message.

# This will clean up the data that is returned to us.

# Once we run it and our data is returned to us we will run response2.choices[0].message.function_call.arguments.

# This will return the two argumnets inside our prompt.

# Those are the loc_origin and the loc_destination.

# As of this point it's not giving us an answer but it is able to extract data from our prompt.

# To get the actual flight data we will have to call in a third party API.

# So far what we have done is created our own function which is working as an API.

# This is the function we created.


#def get_flight_info(loc_origin, loc_destination):
#    """Get Flight Information Between Two Locations"""
#
#    flight_info = {
#        "loc_origin": loc_origin,
#        "loc_destination": loc_destination,
#        "datetime": str(datetime.now() + timedelta(hours = 2)),
#        "airline": "AMS",
#        "flight": "JFK",
#    }
#
#    return json.dumps(flight_info)

# This is the function we will copy into our notebook.

# Then we will copy the response.arguments and pass it into a json.loads() and also assign it a variable of params

# This will bring back our two arguments.

# From here we want to extract some more information.

# We want the origin and destination.

# We will call them like this

# origin = json.loads(response2.choices[0].message.function_call.arguments).get("loc_origin")

# destination = json.loads(response2.choices[0].message.function_call.arguments).get("loc_destination")

# So now we have our parameters which are our origin and destination.

# Now we want to find out the flight details.

# For that we are going to call our chosen_function that we created.

# That will look like this.

# chosen_function = eval(response2.choices[0].message.function_call.name)

# What this eval method will do is get passed a name.

# we will actuall initialize that above our chosen_function.

# What we will do with the eval is, we're going to pass the name of the function.

# This is the function name we will be calling when we call the eval. "get_flight_info"

# The eval isn't actually doing anything, it's just giving us the actual value of our function.

# we can run this in our notebook to check eval(response2.choices[0].message.function_call.name)
# Note: We can do this when the code is in a proper chain and all the cells in our notebook have been checked.

# We can also check the "type" of our functions and other pieces of code by using this formula in our notebook.
# type(response2.choices[0].message.function_call.name)
# Note: We can do this when the code is in a proper chain and all the cells in our notebook have been checked.

# We can also see the difference between using the "type" and the "eval" to find information about pieces of our 
#code in the notebook.

# We can do this by using type("2") in one cell, and then eval("2") in another and see what the results are.

# Notice that type("2") returns a string, and eval("2") returns an integer.

# This tells us that whatever value we pass into the eval we be converted to its original form, which is an integer.

# That was just some useful information.

# Now back to our function. Once we call our function 
# chosen_function = eval(response2.choices[0].message.function_call.name)

# After we call our function we should see something like this 
# flight = chosen_function(**params) # Here is where the parameters are passed, which are the loc_origin and
#loc_destination
# print(flight)

# Once we run this we should get our details.

# So what's happenning is that we actually look at this function as an API.

#def get_flight_info(loc_origin, loc_destination):
#    """Get Flight Information Between Two Locations"""
#
#    flight_info = {
#        "loc_origin": loc_origin,
#        "loc_destination": loc_destination,
#        "datetime": str(datetime.now() + timedelta(hours = 2)),
#        "airline": "AMS",
#        "flight": "JFK",
#    }
#
#    return json.dumps(flight_info)

# So now, we are expecting information from the prompt we created, and we are collecting details of the flight.

# These are the steps we've taken so far.

# First we defined a function

# functions_description = [
#     {
#           "name": "get_flight_info",
#           "description": "Get flight information between two locations",
#           "parameters": {
#               "type": "object",
#               "properties": {
#                    "loc_origin": {
#                        "type": "string",
#                        "description": "The departure airport, e.g. AMS",
#                     },
#                     "loc_destination": {
#                         "type": "string",
#                         "description": "The destination airport, e.g. JFK",
#                     },
#               },
#               "required": ["loc_origin", "loc_destination"],
#           },
#     },
#]

# Next we are linking to our model by using the openai api.

# response2 = client.chat.completions.create(
#    model = "gpt-3.5-turbo-1106",
#    messages = [
#    {
#        "role": "user",
#        "content": user_prompt
#    }
#    ],
#    funtions = function_description,
#    function_call = "auto"
#)

# We aill also be using this function kinda as an API

#def get_flight_info(loc_origin, loc_destination):
#    """Get Flight Information Between Two Locations"""
#
#    flight_info = {
#        "loc_origin": loc_origin,
#        "loc_destination": loc_destination,
#        "datetime": str(datetime.now() + timedelta(hours = 2)),
#        "airline": "AMS",
#        "flight": "JFK",
#    }
#
#    return json.dumps(flight_info)

# Because the model the we linked to will not be able to get us present day flight data, the function
#above will act as our API and return the information we requested.

# Now we will look at the final call in this process

# response3 = client.chat.completions.create(
#    model = "gpt-3.5-turbo-1106",
#    messages = [
#    
#        {"role": "user", "content": user_prompt} # user_propmt is the information we are requesting
#        {"role": "function", "name": response2.choices[0].message.function_call.name, "content": flight} # This
#is where we will get the name of the function we're calling and get the content, which is the flight information.
#    
#    ],
#    funtions = function_description,
#    function_call = "auto"
#)

# From here we can run our response3 and get the present day data we requested.

# And as usual we can get the exact data we want by refining our return by using
#response3.choices[0].message.content

# This is the definition of the funnction calling.

# Learning how to connect Large Language Models to extrenal tools.

# These are steps.

# 1. Define the function
# 2. Define the parameters
# 3. Define the values

# And according to these steps we can get a response from a third party API.



# In this section we will be exploring Langchain

# The very first thing we need to do is import langchain and initialize our Keys againand follow
#these steps below.

# 1. import langchain

# 2. from langchain_openai import OpenAI

# 3. Keys = "sk-proj-NFj08EyWI8laQ9spJBafT3BlbkFJKzklOVqO2yHxrcNMo4lT"

# From here we can start building our code.

# Inside langchain we have different modules, and inside these modules we have different classes.

# Initially we decided that we can look at langchain as a kind of wrapper for openai.

# So now, whatever request we make doesn't go directlt to the openai api, but gets passed through langchain.

# But langchain is not restricted to this api. It has many uses.

# Langchain is very powerful and it is open source.

# Now we will create the object of this OpenAI API, assign it a variable of client, and pass our api key.

# client = OpenAI(openai_api_key = Keys)

# Now we will call our method.

# Our method name will be predict and we will write it like this, client.predict()

# From here we can pass in our parameter, which is going to be a prompt, which is an input, which is what we 
#will be passing to our model.

# Above our method we will define our prompt.

# This is how we will define our prompt.

# With this prompt we will be asking aour model a question.

# prompt = "Can you tell me the total number of countries in Asia"

# Here is our question that we will be asking our model.

# This is called "Zero Short" prompting. 

# Now we can pass our prompt to our method and run it.

# We can also modify the question we ask our model by appending another question after our first question.

# That would look something like this.

# prompt = "Can you tell me the total number of countries in Asia? Can you give me the top 10 countries in Asia" 

# Also, we can use the .strip to remove everything from our return that isn't text, things like slashes and stray
#characters.

# That will look like this.

# client.predict(prompt).strip()

# Lastly we can this to a print() and get back a return in a much more readable format with our information.

# That will look like this.

# print(client.predict(prompt).strip())

# This was a simple demonstration of what we can do with langchain.

# There are many more uses that we will go over later in the course.

from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain_community.llms import huggingface_pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

prompt = PromptTemplate(
    input_variables = ["name"],
    template = "Can you tell me about the YouTuber {name}?"
)

model_id = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(model_id)