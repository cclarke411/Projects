
# coding: utf-8

# In[1]:


import tensorflow as tf
sess = tf.Session()


# # Modeling with TensorFlow
# 
# You're now ready to start building your first model with TensorFlow. To do that, we'll walk through a simple example: a linear model. Remember, Neural Networks are just ensembles of many many linear models, so this is a fine place to start.
# 
# Here it will be worth noting that we also have some shorthand built into TensorFlow to make our code even more readable (hooray for Python!). You'll see what we mean in a moment.
# 
# ## Variables
# 
# To build a model in TensorFlow, we will need another type of input than the nodes we've covered so far. Specifically we'll need TensorFlow variables.
# 
# First we'll implement one below and then describe its uses.

# In[5]:


q = tf.Variable([0], tf.float32)


# Now, a tensor flow variable is somewhere between a constand and a placeholder, with a few of its own added features. Much like a placeholder, it does not have a set value. It does, however, take a default, much like a constant.
# 
# However, the big difference is that variables are not initialized when they are defined. You have to initialize them manually. Like this:

# In[7]:


init = tf.global_variables_initializer()
sess.run(init)


# Now our variables are initialized. Let's try it in a context of a linear model to further explain.

# In[8]:


# Note that our initial value has to match the data type
# so 1 would give an error since it's an int...
b = tf.Variable([1.], tf.float32)
m = tf.Variable([1.], tf.float32)
x = tf.placeholder(tf.float32)

# Implement a linear model with shorthand for tf.add() by using '+'
# and tf.multiply with '*'
linear_model = m * x + b

# New variables means we have to initialize again
init = tf.global_variables_initializer()
sess.run(init)


# Now we can run our model over a given string of values. You can think of our variables as the parameters we want to estimate and our placeholder as the input values for training or testing our model. Let's give it a tensor of some input values and see how that initial value for `b` behaves.

# In[9]:


print(sess.run(linear_model, {x:[1, 2, 3, 4]}))


# So this does what we'd expect with an initial m and b value of 1. For each x value in our tensor it simply added one. But what if we were trying to match a real line?
# 
# For that we need to have training data and a loss function. Again, this is implemented below.

# In[10]:


y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1, 2, 3, 4], y:[.1, -.9, -1.9, -2.9]}))


# Now this uses squared errors as our loss function and we find that our initial values of 1 and 1 seem to be quite poor. When you look at that data perhaps you saw the actual values of m and b?
# 
# We can manually set them with `tf.assign`.

# In[11]:


fixm = tf.assign(m, [-1.])
fixb = tf.assign(b, [1.1])
sess.run([fixm, fixb])
print(sess.run(loss, {x:[1, 2, 3, 4], y:[.1, -.9, -1.9, -2.9]}))


# This gives us a much better error, basically zero. But it's also not that useful. If we already knew the answer we probably wouldn't need to build the model in the first place!
# 
# So how can we use TensorFlow to find the answers for us?
# 
# ## Training in TensorFlow
# 
# We can implement the same optimization framework we did previously: Gradient Descent! This uses the training module within TensorFlow to implement a gradient descent approach to minimizing our loss function. Here's how it works:

# In[12]:


# Set your learning rate in Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# reset values to incorrect defaults.
sess.run(init) 

# Loop for 100 iterations, trying to find optimal values
for i in range(100):
    sess.run(train, {x:[1, 2, 3, 4], y:[.1, -.9, -1.9, -2.9]})

print(sess.run([m, b]))


# Vary the number of loops you run by adjusting the value in range. We find a pretty good solution around 1000 loops.
# 
# This explicitly sets up gradient descent for our given model, thinking about it as a node in the process with adjustable variable values. We do that by establishing an optimizer that is traditional Gradient Descent, and then telling that approach to minimize our loss function.
# 
# With this level of manual control there is a lot you could mess around with. You could make a different loss function. You could make it maximize the loss. You could loop it an exceedingly large or small number of times. The options really are endless.

# You should now be familiar with the mechanisms of modeling in TensorFlow. 
# 
# If you're looking for more, note that everything we've done in this and the previous session is a walkthrough based on TensorFlow's documentation and getting started guide, found [here](https://www.tensorflow.org/get_started/get_started), if you're looking for another variation of this lesson in a slightly varied context.
