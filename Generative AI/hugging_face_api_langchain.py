# In the this next section we will be over Huggingface api with langchain

# This will be a look at the use of langchain in a detailed way.

# This is what we will be going over in this section of the module.

# We're going to start with langchain.

# 1. First we will discuss how to use openai via langchain.

# 2. The second topic will be prompt templates and how to use them.

# 3. The third topic is what are "chains" and how we can use them.

# 4. Next we will talk about agents and what they are.

# 5. The fifth topic we will be going over is the memory.

# 6. The sixth topic will be document loaders.

# After we go over all of these topics we will move into HuggingFace.


# The reason we will be using langchain is because there are limitations with the openai api.

# Here are a few drawbacks.

# 1. OpenAI model is not free.

# 2. A lot of the OpenAI models are not trained up to the present days data, which means it has limited knowledge.

# These are the two main reasons why we will be using langchain.


# Langchain - 

# 1. We can access different LLM models by using different API's.

# 2. We can access private data resources.

# 3. We can access any third party API.

# These are just some of the usecases of langchain.


# Here are a few features of langchain.

# 1. Has chains. (We will discuss chains in a few)

# 2. It can read documents using a document loader.

# 3. It has agent which can access any third party api.

# 4. It access any sort of LLM, whether it's an open api or any other api.

# 5. It can retain deep memory.

# 6. Create different prompt templates.

# So this langchain can do multiple things and perform multiple task.


# Now we will start with the practical implementation of prompt templates.

# We will import them like this

# from langchain.prompts import PromptTemplate.

# Here is how write a template.

# prompt_template_name = PromptTemplate(
#     input_variables = ["city"],
#     template = "can you tell me the capital of {city}"
#)

# First we created an object of the prompt class.

# Then we assigned it to the variable prompt_template_name

# Now, inside of object we will pass some parameters.

# The first parameter we passed is the input_variables.

# And the second parameter we are going to pass is template, which is how we configure our prompt.

# Next, our input_variable parameter will get pass a name, which is ["city"].

# Next, our template will get passed the request we want, which works as a prompt, 
#"can you tell me the capital of {city}"

# Now we can call our method with a .format, and add the city we want named, and run it. The template
#will look like this.
# prompt_template_name.format(city = "New York")

# Now we are going to create a prompt for a country.

# Since we are asking about the country now instead of the city we need a prompt_template to reflect the 
#new request.

# This is our new prompt.

# prompt_template_name = PromptTemplate(
#     input_variables = ["country"],
#     template = "can you tell me the capital of {country}"
#)

# prompt_template_name.format(country = "China")

# Basically, we should have realized that by using this prompt template we can construct the prompt, based on 
#an input_variable.

# Now let's pass our prompt to langchain.

# We will create a variable for both of our prompts

# prompt1 = prompt_template_name.format(city = "New York")

# prompt2 = prompt_template_name.format(country = "China")

# This is how we would call our prompts to our LLM.

# We will call the first prompt like this.

# We will also add a strip() to clean our return data.

# client.predict(propmt1).strip()

# So these prompt templates will be very useful when we are creating applications where we just require a single
#word from the user.

# So to recap, these are the steps we took.

# First we have our PromptTemplate class

# Then, We created our object, prompt_template_name.

# Next we have a method, which is format(). We can also call our method like this format_template(). Whichever one is
#easier for us to remember.



# Now we will be going over the "Agent" in langchain.

# What is an agent?

# The agent is used in langchain to call any third party tool.

# This is the simple definition of the agent.

# Basically the agent collects data from the third party tool and provides it to us.

# Now let's start understanding our agent.

# We will start by asking a questions of our chat-gpt.

# This the propmt that we will use to ask our model a question.

# prompt = "Who won the last superbowl?"

# Next we will use the client predict and pass our prompt.

# And as usual we will add .strip() to clean up the return data.

# client.predict(prompt).strip()

# Because using this method will return us outdated data(because the model is only trained up until a certain point),
#this is where the agent will come into play.

# The agent will extract the information from the third party api.

# To extract real time information we will be using the serp api.

# By using the serp api we will call the google search engine.

# And we will extract the information in real time.

# Now let's go over the steps we will be taking to make this happen.

# First we must install google search.

# Next we will create our serp api key.

# We need this key to access the api.

# We can get this key from google search in our browser.

# What is the serp api?

# SerpApi is a real-time API to access Google search results. We handle proxies, solve captchas, and parse all 
#rich structured data for you.

# Once we get inside the serp api website, we will navigate to the google api documentation section.

# This section will have all the api's and the information the api's.

# We can go over the different options to get familiar with everything.

# But to get our API key we will hit the API key button at the top of the page.

# By using our generated api key we can access all of the api's listed on the website.

# After we put our api key in our notebook, we can start bringing over our imports.

# These are the imports we need.

# from langchain.agents import AgentType

# from langchain.agents import load_tools

# from langchain.agents import initialize_agent

# from langchain_openai import OpenAI

# Next we will create a client.

# This is how we will write our client.

# Next we will create an object of the load_tools class.

# We will do that by writing this line of code.

# tools = load_tools()

# Now we have to keep in mind that we have to pass this as a parameter to ensure that our created function works
#properly.

# This is how our line of code should look.

# tool = load_tools(["serpapi"],serpapi_api_key=OPENAI_KEY,llm=client)

# Now we have created our tool.

# Next we will create our agent type.

# We will do that by creating an object of the initialize_agent

# This is how we will create the object.

# initialize_agent()

# And this is how we create our agent.

# agent = initialize_agent()

# Just like with the previous created object, we will need to pass in parameters for our object to work propperly.

# This is how it will look.

# agent = initialize_agent(tool,client,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)

# In this object we mentioned four parameters, tool, client, agent, and verbose.

# Now this is our agent.

# Now we will run it and this is how we do that.

# agent.run("")

# This run meythod will take a parameter.

# The parameter will be the question we asked .

# Note: I did not realize that we are still connected to the OpenAI website so we have to wait until we have
#our serp api connected to an open source model to fully engage with a model of our own.

# But for now, we will install wikipedia.

# The next thing we have to do is load the tool.

# We can do that using this line of code.

# tool = laod_tools(["wikipedia"], llm = client)

# Next we can do the agent.run() again.

# But we still need to pass in our question as a parameter, so this is what our code will end up looking like.

# agent.run("What is the federal reserve bank?")

# So in a recap, using the serpapi, we can access real time data.

# This was a simple and basic walkthrough and application of the agent.



# It is recommended that we visit python.langchain to read over the documentation located in "Get Started" and get a 
#better understanding of the langchain enviornment.

# Topics to read right away. We will update this list as we go on.

# 1. Agents
# 2. Chains


# The next practical code we are going to use is the chains.

# Chains - Central to langchain, is a vital component known as langchain chains, forming the core connection among
#one or several large language models (LLM's). In ceratin sophisticated applications, it becomes necessary to chain
#LLM's together, either with each other or with other elements.

# This was just a brief overview of Chains. We can find much more detailed descriptions on the langchain website
#in the documents section.

# Basically a chain is just a connection of several components.

# Using LLM's in isolation is fine for simple applications, but more complex applications require chaining LLM's,
#either with each other or with other components.

# Now we can start with a basic of example of chaining.

# First we will start with a client.

# client

# Next we will import the prompt template

# from langchain.prompts import PromptTemplate

# And by using this prompt template we are going to create our propmt.

# prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}")

# Next we will create a prompt for the product that company makes.

# prompt.format(product="Wine")

# This will become our new prompt.

# prompt = prompt.format(product="Wine")

# Next we will import the LLMChain like this

# from langchain.chains import LLMChain

# Now we will create an object of this LLMChain.

# Inside this LLMChain we will pass in our client as a parameter, and our prompt.

# LLMChain(llm=client.prompt=prompt)

# This is our llm chain. We have connected both our llm and our prompt

# To this we will add the object we created, which is chain.

# chain = LLMChain(llm=client,prompt=prompt)

# Next we will run the chain with a parameter of Wine like this.

# chain.run("Wine")

# When we are connected to our source model, we will have a good name for our company returned.

# Lastly we will add a .strip to the run, like this.

# chain.run("Wine").strip()

# Note: The instructors companys name was "Vintage Vines Winery"

# This is a simple example of chaining.



# Now we will look at another example.

# First we will add our prompt template.

#prompt_template=PromptTemplate(
#    input_variables=["cusine"],
#    templates="I want to open a restaurant for {cusine} food, suggest a fancy name for this"
#)

# Now we can find out the prompt template by just running the prompt template.

#prompt_template

# From here we will make a chain.

# Thiis is how we will do this.

# chain=LLMChain(llm=client, prompt=prompt_template)

# Here we will combine two things. First is client, and the second is the prompt_template.

# Then we will run our chain, like this, where we will also pass a parameter to our run method.

# chain.run("chinese")

# This will give us an answer.

# As usual we can further clean this data adding a .strip() to the end of our run method which will end looking
#like this.

# chain.run("chinese").strip()

# Note: We can pass in any type of cuisine that we would like to get a good name for. We just used "chinese"
#as an example.

# Now let's look at another aspect of example 2.

# We will add a parameter of verbose to our chain method.

# This parameter will allow us to check all of the details happenning in the background.

# We will set the verbose equal to true and our chain will now look like this.

# chain=LLMChain(llm=client, prompt=prompt_template, verbose=True)

# Now we will run the chain again and this time we should a lot more details as our request is returned.

# Now let's explore one more concept.



# Here we will look at if we want to combine multiple chains and set a sequence, for which we can use a simple
#sequential chain.

# Now let's try to use the simple sequential chain to better understand what it is.

#prompt_template_name=PromptTemplate(
#    input_variables=["startup_name"],
#    template="I want to start a startup for {startup_name}, suggest a good name for this"
#)
#
#name_chain=LLMChain(llm=client,prompt=prompt_template_name)
#
#prompt_template_items=PromptTemplate(
#    input_variables=["name"],
#    template="suggest some strategies for {name}"
#)
#
#strategies_chain=LLMChain(llm=client,prompt=prompt_template_items)

# So let's try to understand this piece of code step by step.

#prompt_template_name=PromptTemplate(
#    input_variables=["startup_name"],
#    template="I want to start a startup for {startup_name}, suggest a good name for this"
#)

# In this first section we are saying that we want to start a startup, and we want a suggestion for a good name.

# In the next step we created a chain using a model(which is our client) and our prompt_template_name.

#name_chain=LLMChain(llm=client,prompt=prompt_template_name)

# This is the first section of chain from our multi chain.

# Then we have the beginning of the second section of our multi chain with the start of our second prompt.

#prompt_template_items=PromptTemplate(
#    input_variables=["name"],
#    template="suggest some strategies for {name}"
#)

# In this prompt we want a suggestion for some strategies for selecting a good name.

# Now we will combine these two sections of chain into one multi chain using a simple sequential chain.

#name_chain=LLMChain(llm=client,prompt=prompt_template_name) = chain 1

#strategies_chain=LLMChain(llm=client,prompt=prompt_template_items) = chain 2

# We will import a simple sequential chain

# from langchain.chains import SimpleSequentialChain

# After we run it we will create an object of it.

# This is our object we created.

# chain = SimpleSequentialChain()

# Now inside this method we will pass a parameter.

# This is what our method will look like after we pass our parameter.

# chain = SimpleSequentialChain(chains=[name_chain,strategies_chain])

# So automatically it will call to the startup_name (see lime 435), and generate a startup name.

# Then it will give that name to the template we created (see line 450).

# This is an example of automatic chaining.

# Now we will see how this is possible.

# We will create another object by caling a method.

# This is the method we will be calling.

# chain.run()

# Then we will pass in a parameter for this method.

# We will pass in the startup we want to open, which is artificial intelligence. This is how our chain will
#look now.

# chain.run("artificial intelligence")

# Note: the process worked, but keep in mind that we were unable to generate a return because we are still using
#the OpenAI model which we don't have access to, but the actual process and code is working.

# Now to our strategies back in the form of a sorted number list, we can pass it as a parameter to a print function.

# print(chain.run("artificial intelligence"))

# This is a simplesequential chain.



# Now we will be going over the understanding of a sequential chain.

# So far we've been looking at the simple sequential chain. Now we will be looking at the sequential chain.

# The sequential chain has more power than the simple sequential chain, where we can keep the sequence of the
#different prompts.

# We will use this code to examine sequential chaining.

#client = OpenAI(openai_api_key=OPENAI_KEY, temperature=1.2)

#prompt_template_name=PromptTemplate(
#    input_variables = ["cuisine"],
#    template = "I want to open a restaurant for {cuisine}, suggest a fancy name for it"
#)

#name_chain=LLMChain(llm=client, prompt=prompt_template_name, output_key="restaurant_name")

# Now we will focus on the output key.

# And our output key consists of our "restaurant name"

# Now let's see how we will implement this output key in our notebook.

# Before we do that we will define one more prompt_template.

# This is our new prompt_template.

#prompt_template_items=PromptTemplate(
#    input_variables = ["restaurant_name"],
#    template = "Suggest some menu items for {restaurant_name}"
#)

#food_items_chain=LLMChain(llm=client, prompt=prompt_template_items, output_key="menu_items")

# Whatever name we restaurant_name we get from the first prompt will be passed as the restaurant_name for the 
#second prompt.

# Now we will import the SequentialChain, Not to be confused with the simple sequential chain.

# We will import it like this from langchain.chains import SequentialChain

# Now we will run the final piece of code to see the answer.

# This is the final code that will go into our notebook.

#chain = SequentialChain(chains = [name_chain, food_items_chain],
#    input_variables = ["cuisine"],
#    output_variables = ["restaurant_name", "menu_items"]
#)

# Now we have the object we created, which is SequentialChain.

# Then we have as parameters what we want to chain.

# Then we have our input variable, which is cuisine, followed by our output variables which are restaurant_name
#and menu_items.

# Now we can run and get our final response.

# We will run it in our notebook by calling our method chain, and specify that the cuisine we want a response for
#is American. That will end up looking like this.

# chain({"cuisine":"American"})



# Now we will go over another concept that we will use in the future, which is the Document Loader.

# In this section we will go over the steps that will allow us to read any sort documents using langchain.

# These are the Document Loaders

# 1. CSV
# 2. File Directory
# 3. HTML
# 4. JSON
# 5. Markdown
# 6. PDF

# We can get a detailed breakdown of the document loaders if we google Document Loaders Langchain documentation.

# We will begin by going over the PDF.

# First we will install PDF using py pdf

# Next we will use this command to import it into our notebook

# from langchain.document_loaders import PyPDFLoader

# Next we will create an object of this particular class.

# This is how we will we create our object for our notebook.

# loader = PyPDFLoader("C:\Users\alpha\Downloads\Brian J Gates Resume.pdf")

# Note: The PDF we are using will come from our own files. Notice that I am using a PDF from my personal device.

# Also note that we must add an r to the beginning of our parameter in order to avoid an error. That will look 
#like this.

# loader = PyPDFLoader(r"C:\Users\alpha\Downloads\Brian J Gates Resume.pdf")

# Notice the difference between lines 605 and 612.

# Now we can check inside our loader to see what we have by simply running loader in our notebook.

# Next we will run pages = loader.load_and_split() in our notebook.

# After that we will call pages and this should bring back the information inside our pdf.

# This as well as the other document loaders is very important for collecting any form of data.

# So if we want to read any document with langchain we can do it because of these loaders.



# In this section we will be going over the Memory concept of langchain.

# Before we go into the practicals of Memory we will go over the concepts.

# What is a Memory inside of langchain and how can we use it.

# We will be exploring this concept by going over some code that will help us breakdown this topic.

# First we will start by importing our PromptTemplate to our notebook again

# We will also import the LLMChain to our notebook.

# This is another import that we will need in our notebook, OpenAI.

# The next thing we will do is create our client.

# This is how we will implement our client in our noteboook

# client=OpenAI(openai_api_key=OPENAI_KEY)

# Now that we have our imports and our client we can move forward with the understanding of the Memory concept.

# The first thing we want to do is create a prompt template.

# This is the prompt template we will use.

#prompt_template_name=PromptTemplate(
#    input_variables = ["product"],
#    template = "What is a good name for company that makes {product}"
#)

# So now we have our prompt template.

# Now we are going to call our model, but not directly, we will use the llm chain.

# This is where we will connect two componenets.

# The first component will be our client object that we created.

# Our second component will our prompt template.

# Let's now implement our llm chain.

# We will do that by first creating an object of the llm chain, which we will keep inside a chain variable.

#chain = LLMChain()

# Next we will pass in our parameters, which will be llm=client and prompt=prompt_template_name.

# This is what we will have now.

# chain = LLMChain(llm=client, prompt=prompt_template_name)

# Now we have our chain.

# Next we will call our run method of our chain and passs in the question we want to ask our model.

# That will look like this.

# chain.run("colorful cup")

# Next we can add this to a print and add a .strip() to clean up our return.

# This is what we will have 

# print(chain.run("colorful cup").strip)

# Now to go over the memory concept we will use the same prompt by copy and paste.

#prompt_template_name=PromptTemplate(
#    input_variables = ["product"],
#    template = "What is a good name for company that makes {product}"
#)

# We will call the run again, but this time we will ask about drones, instead of colorful cups.

# Now, if we want to call the parameter chain memory we will use this code.

# chain.memory

# But this will not return anything for us so we have to modify it like this.

# We can check what's inside the chain.memory by using a type(chain.memory)

# This will return a nonetype, letting us know that we don't have anything in that particular chain.memory.

# So to move forward we will start by discussing Conversation Buffer Memory.

# Here, we want to save converstion memory.

# There are 4 topics we want to go over in this section to complete this process.

# 1. The first topic is going to be about whatever conversation we have with our model. This will store our
#conversation for us. It will sustain the entire memory of the conversation.

# 2. So we just need to keep in mind one parameter, which is memory. This will help us remember all the previous
#conversations we had regarding this model.

# How can we do this.

# First we wil focus on the memory from langchain by importing it, then we will see the same thing from the
#persceptive of the documentation.

# This is how we will import it.

# from langchain.memory import ConversationBufferMemory

# Now we will create an object of the ConversationBufferMemory class and store it inside of a variable
#called memory.

# This is what it will look like and how we will run it in our notebook.

# memory = ConversationBufferMemory()

# Now we will create a prompt template (we will reuse the one we already have for this example.)

#prompt_template_name=PromptTemplate(
#    input_variables = ["product"],
#    template = "What is a good name for company that makes {product}"
#)

# Now we will use the LLMChain method and pass in the our 2 parameters, llm=client and prompt=prompt_template_name,
#which will look like this LLMChain(llm=client, prompt=prompt_template_name).

# Now if we want to retain the conversation we will add an addition parameter to our method, which will be
#memory=memory, and we will store it all in a variable called chain.

# Now this is what we have.

# chain = LLMChain(llm=client, prompt=prompt_template_name, memory=memory)

# Now we have created our new object with memory and we can run it.

# This is how we wil run it as usual.

# chain.run("").strip()

# Let's say that we asked the api for a bunch of different names for a bunch of different companies (because we can't
#use the OpenAI right now). We should be able to recall each and every one of the original returns in this
#particular conversation by using this command.

# chain.memory

# This is all thanks to the ConversationBufferMemory

# Let's try to understand a little better what's going on.

# If we use chain.memory.buffer, this will bring back our entire conversation, but in a more readable format.

# Further more, if we add this as a parameter to the print() we can get a much cleaner return of our information,
#and it will be returned as an actual conversation between us and our api.

# print(chain.memory.buffer)



# Now we are able to maintain previous conversations.

# Now we will go over a few more concepts.

# We will be introduced to a new concept called ConversationChains




#   ConversationChains

# Conversation buffer memory grows endlessly - Meaning we can sustain whatever is in that particular conversation.

# Just remember last 5 Conversation Chains - This is if we want to only remember the last 5 conversation chains.

# Just remember last 10-20 ConversationChain - This is if we want to remember the last 10-20 conversation chains.

# We understand this better by working on some examples.

# First we'll import ConversationChain from langchain.chains

# 



# Here we will look at the ConversationBufferWindowMemory

# 