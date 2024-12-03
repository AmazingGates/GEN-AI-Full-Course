# What Is Generative AI

# Generative AI generates new data based on training samples. Generative models can generate Images, Text, Audio, Videos,
#and other types of data output.

# So Generative AI is a very huge topic, consisting of two main segments -
# - Generative Image Models
# - Generative Language Models

# LLM's actually fall into the Generative Language Models category.

# Now to backtrack a little, we will be going looking at the architecture of a GAN.

#   _____________             _________                                    _____________________
#  |             |           |         |                                  |                     |
#  | Real Images |---------> |  Sample |\                               / |  Discriminator Loss |
#  |_____________|           |_________|  \         _______________   /   |_____________________|
#                                           \      |               |/
#                                             ---> | Discriminator |
#   _____________             _________     /      |_______________|\      _____________________
#  |             |           |         |  /                           \   |                     |
#  |  Generator  |---------> |  Sample |/                               \ |    Generator Loss   |
#  |_____________|           |_________|                                  |_____________________|

# So in the GAN, we have two main conponents.

# The first conponent is the generator.

# The second conponent is the discriminator.

# These two conponents work as a neural network, where we will be passing in data.

# From here we can pass in Real Image data, and Generator(Synthetic) data as samples into the discriminator.

# So the Generator and the Discriminator are the two main conponents inside the GAN architecture.

# And the Generator and Discriminator operate as a Neural Network.

# This Gan Falls under Generative AI.

# The Generative Image Model to be more precise.

# Note: We should keep in mind that we can also generate images with powerful LLM's ae well.

# We will go over a few models regarding this process of generating images using LLM's like Delhi, for example,
#in the future.

# But back to the Generative Image Model, which we use with image to image generation.

# The Generative Langauge Model is used for text to image generation, and text to text generation.

# In past years, image to image generation was primarily handled by GAN's, but as we stated, now a days it can 
#be performed with powerful LLM's.

# Now let's look at an example below.


# Questions ------> Cat Image ------> Responses

# In the example, the Cat Image is representing a generative model, where we are giving a prompt, or giving a 
#question, and we get a purpose, or a response.

# So here we are talking about generative models, not specifically LLM's.

# Remember that we think of Generative Models as a super set of Generative AI.

# GAN's and LLM's are under the umbrella of Generative AI.

# Now back to our example, where we are givig an input and getting an output.

# Regarding the LLM's, the inputs are generally known as Input Prompts, and the outputs are known as Outptut Prompts.

# So the image of the Cat can be thought of as a Geneative model, or LLM model.

# So we are passing an input prompt and we are getting an output prompt.

# Note: The prompt is very important.

# Now let's look at this list below to get a better understanding of what this field is made up of.

# Machine Learning is the subset of artificial intelligence.
# Deep learning is the subset of Machine Learning.
# Generative AI is a subset of Deep Learning.

# Now remember, we can think of Generative AI as a superset, and inside Generative AI there are many tpoics and concepts.

# The two main topics we will be going over in Generative AI are the G A N and the L L M

# - G A N - Which is our generative adverserial network

# - L L M - Which is our large language model.

# Inside the Generative AI we will have various task.

# These task include but are not limited to

# 1 - Image to Image generation
# 2 - Text to Text generation
# 3 - Image to text generation
# 4 - Text to Image generation

# The image to image is a feature of the GAN
# The text to text can be performed by various models, but is primarily a feature of the LLM
# The image  to text can also be performed by various models, but it is also primarily a feature of the LLM
# And these days the task of text to image is carried out by LLM

# So LLM's are  able to perform various task, some heterogeneous and some homogeneous.

# Now back to the Generative AI.

# Where does the Generative AI actually exist?

# It exist in a subset of DeepLearning.

# Now let's go over the complete timeline of the LLM

# First we will go back into the R N N, which is a type of Neural Network.

# Remember that this is basically when we have a feedback loop, which takes the output and loops it back into the 
#hidden state as input.

# Note: There are different types of R N N's, or advanced architecture in terms of this R N N.

# Also note that the R N N is for the Short Term Memory. This means that we can not retain a long sentences or a 
# a large sentence using a R N N.

# Another type of R N N is the LSTM.

# LSTM - With the LSTM we are able to retain long and large sentences because it is Long--Short Term Memory.

# This means that it is for the short term dependency and the long term dependency.

# If we look at the architecture of the LSTM we will see that along with the timestamp, we would have the settle
#state.

#
# Settle state = ------------------- # This will maintain our longterm dependency
#
# Time Stamp   =  []   []   []   [] 

# We will have connections between the settle state and time stamps that are called gates.

# These 3 gates will be the Forget gate, the Input gate, and the Output gate

# This will look something like this.


#                    F    I    O
# Settle state = ------------------- 
#                   | |  | |  | |
# Time Stamp   =  []   []   []   [] 


# These are the 3 gates inside our LSTM for sustaining long term dependency, or maintaing long sentences.

# Now we will look at one more updated version of the LSTM, which is known as a GRU.

# G R U - The G R U was created in 2014 and it was taken specially from the LSTM. Inside the G R U we won't find the
#concept of the settle state. We'll just have two gates. The save gate, and the update gate. This is considered an
#an updated version of the LSTM

# These 3 architectures/Models (R N N, LSTM, G R U) were very poupular from 2018 thru 2020, and by using these particular 
#architectures we are going to process sequence data.

# One concept to keep in mind is sequence to sequence mapping, and for that only, we are using these particular
#architectures.

# We have different types of mapping techniques for other scenarios. We will list them below.

# 1) One to One 

# 2) One to Many

# 3) Many to One

# 4) Many to Many

# We can actually implement these mapping sequences by using the RNN, LSTM, and GRU.

# If we're talking about these particular sequences we can definitely create various applications by using our 
#three models, but these sequences come with restrictions.

# Some of the applications for these sequences include

# 1) One to One 

# 2) One to Many - Can be used for image capturing

# 3) Many to One - Can be used for sentiment analysis

# 4) Many to Many - Can be used for language translation.

# So there are various applications of these sequences.

# Now, if we're talking about the sequence to sequence mapping, we can definitely implement it by  using these 
#particular architectures, RNN, LSTM, GRU.

# The problem we would have for example, if we had an input of 5 words, we would need an output of 5 words as well.

# So it'sa fixed length for input and output.

# Using sequences like many to many will cause issues because of the fixed length. Whatever number of inputs we 
#enter, in terms of the many to many we are talking about, we need to have the same number outputted.

# Because of issues like this, a solution was created in the form of sequence to sequence learning, which gave us
#the concept of encoder - decoder.

# In the encoder - decoder, the first segment is the encoder segment, and the second segment is the decoder segment.

# And inbetween we actually have the context vector.


#                                            ______________________
#                                           |                      |
#                                           |                      |
#   _________________             __________|                      |
#  |                 |           |          |                      |
#  |                 |           |          |______________________|
#  |                 |___________|                 Decoder
#  |                 | Context Vector
#  |_________________|
#    Encoder

# This model specifies that whatever information that we pass to the decoder from the encoder is passed through the
#context vector.

# However, in the first versions of the encoder - decoder, the architecture was unable to handle longer sentences with
#more than 30 through 50 words. 

# It was not able to sustain the context.

# What makes up the encoder and decoder?

# Inside of our encoder - decoder archutecture will find the RNN, LSTM, and the GRU.


#                                            ______________________
#                                           |                      |
#                                           |        RNN           |
#   _________________             __________|       LSTM           |
#  |                 |           |          |        GRU           |
#  |      RNN        |           |          |______________________|
#  |     LSTM        |___________|                 Decoder
#  |      GRU        | Context Vector
#  |_________________|
#    Encoder

# Now back to the issue that we were discussing about the problem with the many to many sequence mapping, regarding
#classical memory, where there is a restriction that says that the number of values entered into the input has
#to be the same number of values that come out of the output.

# Using the encoder - decoder architecture we can solve that particular issue.

# But here also we were having the issue where the encoder - decoder was unable to handle sentences with more than 
#30 through 50 words.

# This is where our next concept comes into play.

# This is the concept of attention.

# Basically, attention was another way of mapping.

# So let's say we have 5 words in the encoder, attention would try to map each of those words and look for a 
#corresponding word in the decoder.

# In essence, trying to match the words in the input to the words in the output.

# And because of this it was able to make predictions.

# So whatever longer sentences we had as input in the encoder, it was able to predict if there was a match for it
#in the output.


#                                            ______________________
#                                           |                      |
#                                           |        RNN           |
#   _________________             __________|       LSTM           |
#  |                 |           |          |        GRU           |
#  |      RNN        |           |          |______________________|
#  |     LSTM        |___________|                 Decoder
#  |      GRU        | Context Vector
#  |_________________|        |
#    Encoder                  |
#                             |
#                         Attention


# In later versions of the encoder - decoder, it became possible to map the inputs to outputs without using
#RNN, LSTM, or GRU.

# This means that in order to use the encoder - decoder, all we needed was the attention mechanism.

# So for example, if we were passing an input into the encoder with a text longer than 30 or 50 words, the 
#attention mechanism was all we needed to generate an output.


# Now we will touch on the topic of Discrimintive vs Generative Models

#   Discriminative vs Generative Model

#   ___________
#  |           |
#  | Cat Image |         _________________
#  |___________|\       |                 |
#                \______| Discrimintaive  |
#   ___________  /      |    Model        |--------- CAT | DOG
#  |           |/       |_________________|
#  | Dog Image |
#  |___________|


#   
#   _________________             ________________                _____________
#  |                 |           |                |              |             |
#  |                 |           |   Generative   |              |     Cat     |
#  |                 |---------> |     Model      |------------> |    Image    |
#  |_________________|           |________________|              |_____________|
#  

# What are the differences between the Discriminative and Generative Model.


# Discriminative Model
#   _________________             ________________                   Rock
#  |                 |           |                |                  /
#  | Music Data /    |           |  Discriminator |                 /
#  | Type of Music   |---------> |  DL Model      |-----------------------> Classical
#  |_________________|           |________________|                 \
#                                                                    \ 
#                                                                     Romantic


#   Generative Models
#   _________________             ________________                _____________
#  |                 |           |                |              |             |
#  |                 |           |   Generative   |              |     New     |
#  |                 |---------> |     Models     |------------> |    Music    |
#  |_________________|           |________________|              |_____________|
#  


# When we're talking about the Discriminative Model, we're basically talking about everything we've learned so far
#in  classical machine learning and deep learning.

# Any Classification based model.

# For example, let's say we're talking about the RNN as our discriminator model, we would be traing our model on 
#specific data.

# Music data would be our Data, and Type of music would be our input.

# We would be performing a supervised learning.

# We would be performing a supervised learning by using the Recurrent Neural Network.

# This is a classical model.

# We have other classical models as well, like the machince learning name bias, or maybe some other model.

# So, we have our RNN model, and we are going to train that model using the supervised machine learning.

# This is going to get passed a specific type of data to this particular model that will in return produce specific
#types of outputs.

# Outputs like Rock music, Classical music or Romantic music.

# So we are passing music to our model and it is going to predict something like the genre.

# This is the discriminative model.


# Now, when we're talking about a generative model, it's a little different than the discriminative model.

# How are they different?

# First of all, if we're talking about a generative model, the training process is a little different, If we're
#talking about the Large Language Model.

# The process of training the LLM is a little different compared to the discriminative model.

# So we are talking about Generative models. We are passing our inputs into this model and getting an output.

# But How?

# The first step, we will be using unsupervised learning.

# IN the second step, we will be using supervised fine tuning.

# And in the third step we will be using reinforcement learning.

# Reinforcement learning has recently been used for chat gpt models, but before that, whatever LLM they had created,
#they had created on a large amount of data.

# So for that, first they had performed the unspervised learning, then they had performed the supervised fine
#tuning.

# So because of that, that model would be able to understand each and every pattern that was there inside the data.

# Because of that it was able to generate an output.

# So basically, the generative model takes input, fine tunes and trains it using various steps, and then produces new 
#data based on the data it got from the model. 

# This means that it is generating new data.

# And this is basically the difference between models.

# The Discriminative model is a classical model, like supervised learning, where we use RNN.

# The Generative model uses various steps for training and uses LLM's, and is responsible for generating new data.



# Now we will look at a few more examples dealing with this concept.



#            Unsupervised Learning                              Supervised Leraning
#     ________________________________________          _____________________________________
#    |    o o o o                             |        |                        o / o        |
#    |   o . . .  o        o o o o o o o o    |        |                      o / o          |
#    |  o  . . . . o     o      .    .     o  |        |                 o    / o            |
#    |  o  . . . . o    o   .  .   .        o |        |                  o / o              |
#    |   o  o  o  o     o     .   .    .     o|        |               o  /  o               |
# X2 |                  o   .         .      o|    X2  |              o /o                   |
#    |                   o        .     .   o |        |            o / o                    |
#    |     o   o  o  o   o o o o o o o o o    |        |         o  /o                       |
#    |   o  .     .    o                      |        |        o /  o                       |
#    |  o      .     .   o                    |        |   o    /   o                        |
#    | o  .   .    .  .   o                   |        |     o/  o                           |
#    | o      .    .    .o                    |        |  o /  o                             |
#    |   o  o  o  o  o  o                     |        | o/  o                               |
#    |________________________________________|        |/____________________________________|
#                     X1                                                 X1


#   Clustering:                                            Classififcation and Regression:
#   K-Means 
#   DBScan


# Generative AI is a subset of deep learning and Generative models are trained on huge amounts of data.

# While training the genertaive model we don't need to provide the label data. 

# It is not possible when we have a huge amount of data, So let's just try and see the relationship between the
#distribution of the data.

# In generative AI we give unstructured data to the LLM Model for training purposes.


#     _____________________________________________________________________________
#    |     [Dog Image]                    [Dog Image]                o / o        |
#    |         |___________o o o o o o o o _______|                o / o          |
#    |                   o      .    .     o                  o    / o            |
#    |                  o   .  .   .        o                  o / o              |
#    |                  o     .   .          o  [Cat Image] o  /  o               |
#    | [Cat Image]      o   .         .      o      |______o /o                   |
#    |       |           o                  o            o / o                    |
#    |     o   o  o  o   o o o o o o o o o            o  /o                       |
#    |   o  .     .    o                             o /  o                       |
#    |  o      .     .   o                      o    /   o    [Dog Image]         |
#    | o  .   .    .  .   o____                   o/  o________|                  |
#    | o      .    .    .o     |               o /  o                             |
#    |   o  o  o  o  o  o  [Cat Image]        o/  o                               |
#    |                                     o / o                                  |
#    |      Generative                 Discriminative                             |
#    |____________________________________________________________________________|


# So in this model, we first perform the unsupervised learning for our generative / clustering, and then perform 
#the other various steps in the process.

# Then we will perform supervised learning for the discriminative / classification and regression.


# Now we will look a little closer at LLM's

# Large language models (LLM's) are foundational machine learning models that use deep learning algorithms to
#process and understand natural language. These models are trained on massive amounts of text data to learn patterns
#and entity relationships in the language.

# It is a language model which is responsible for performing task such as text to text generation, text to image
#generation and image to text generations.

# A large language model is a trained deep learning model that understands and generates text in a human like
# #fashion.

# LLM's are good at understanding and generating human language.

# So how do we use LLM's?

# Let's look at the example below.


#   The         garden         was         full         of         beautiful         [flowers]
#    |            |             |            |           |            |                  |
#    |            |             |            |           |            |                  |
#    |            |             |            |           |            |                  |
#    |            |             |            |           |            |                  | 
#    |            |             |            |           |            |                  |
#    |            |             |            |           |            |                  |
#    |____________|_____________|____________|___________|____________|_________________LLM


# Why do we call it a large language model?

# Because of the size and complexity of the neural network as well as the size of the dataset that it was trained
#on.


# Researchers started to make these models large and trained on huge datasets.

# They started showing impressive results like understanding complex natural language and generating language 
#more eloquently than ever.


# What makes LLM's so powerful?

# In the case of the LLM, one model can be used for a whole variety of task like: -

# - Text Generation
# - Chatbot
# - Summarizer
# - Translation
# - Code Generation
# - And Much More...

# LLM is a subset of deep learning and has some properties that merge with Generative AI.


# LLM's Model Architecture

# Large language models are based on transformers, a type of neural network architecture invented by google.


# These are a few milestones in the large language model -

# - Bert: Bidirectional Encoder Representations from Transformers was developed by google.

# - GPT: GPT stands for "Generative Pre-trained Transformer". The model was developed by OpenAI.

# - XLM: Cross-lingual Language Model pretraining by Guillaume Lample and Alexis Conneau.

# - T5: The Text To Text Transformer, it was created by google.

# - Megatron: Megatron is a large, powerful transformer, developed by the Applied Research team at Nvidia.

# - M2M 100: Multilingual encoder-decoder (seg-to-seg) model researchers at facebook used.

# These were all milestones of the LLM.


#   Transformer Tree

#                   _______________________
#                  |       Transformer     |
#                  |                       |
#              --- |  [Encoder]-[Decoder]  |---
#              |   |_______________________|  |
#              |               |              |
# [DistBert]-[Bert]           [T5]          [GPT]
#              |               |              |
#          [RoBerta]         [BART]        [GPT-2]-[CTRL]
#              |               |              |
#    [XLM-R]-[XLM]         [M2M-100]       [GPT-3]
#              |               |              |
#           [ALBERT]       [BigBird]       [GPT-Neo]-[GPT-J]
#              |                              |
#           [ELECTRA]                         
#              |
#           [DeBerta]

# This transformer tree lets us know that some models use the encoder, some models use the decoder, and some use both.

# These are just a few examples, there are other models as well.


#   OpenAI Based Models

# Models           Description

# GPT-4        -  A set of models that improve on GPT-3.5 and can understand as well as generate natural language
#or code.   

# GPT-3.5      -  A set of models that improve on GPT-3 and can understand as well as generate natural language
#or code.

# GPT base     -  A set of models without instruction that can understand ae well as generate natural language or
#code.

# Dall-E       -  A model that generate and edit images given a natural langage prompt.

# Whisper      -  A model that can convert audio into text.

# Embeddings   -  A set models that can convert text into a numerical form.

# Moderation   -  A fine-tuned model that can detect whether text may be sensitive or unsafe

# GPT-3 Legacy -  A set of models that understand and generate natural language.

# These are the different models we may come across on the OpenAI website.

# GPT is the most commonly model used to today out of the group, and it can perform any sort of task related to
#generation.


#       Other Open Source Models -

# -  Bloom

# - Llama 2

# - PaLM

# - Falcon

# - Claude

# - MPT-30B

# - Stablelm

# - And many more...

# Unlike the OpenAI models, these other open source models may be free.

# We will be going over the process of utilizing these models as well.


#       What can LLM's be used for?

# - Text Classification

# - Text Generation

# - Text Summarization

# - Conversation AI, like Chatbot / Question Answering

# - Speech Recognition and Speech Identification

# - Spelling Corrector

# - And many more use cases.



# Other topics we will discussing are things like prompt designs.

#       Promt Designs -

# - All the text that we will feed into an LLM as input is called a prompt, and this whole art is known as Prompt
#Design, which is about figuring out how to write and format prompt to text to get LLM's to do what we want.

# Examples of promts -

# [It's raining cats and...]

# [I have two apples and I eat one, I'm left with...]

# We will dive deeper into this subject as the course progresses, this is just an over look.



# We will also be going over topics like how gpt was trained.

#       How was ChatGPT Trained?

# - Internally using LLM which is gpt-3.5 or gpt-4.

# - It was trained on a large amount of data which is available all over the internet.

# These are usually the steps of training.

# 1. Generative Pre-Training
# 2. Suervised Fine-Tuning
# 3. Reinforcement Learning

# This is another topic that we will be diving further into once we get deeper into the course.