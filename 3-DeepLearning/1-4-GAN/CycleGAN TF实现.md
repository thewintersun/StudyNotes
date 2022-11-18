## Cycle GAN TF Implementation 

代码地址: https://github.com/hardikbansal/CycleGAN/

博客地址： https://hardikbansal.github.io/CycleGANBlog/

### Network Architecture

![Model Architecture](https://hardikbansal.github.io/CycleGANBlog/images/model.jpg)





------

![Model Architecture 1](https://hardikbansal.github.io/CycleGANBlog/images/model1.jpg)

### Building the generator

High level structure of Generator can be viewed in the following image.

![Model Architecture 1](https://hardikbansal.github.io/CycleGANBlog/images/Generator.jpg)



The generator have three components:

1. Encoder
2. Transformer
3. Decoder

Following are the parameters we have used for the mode.

```python
ngf = 32 # Number of filters in first layer of generator
ndf = 64 # Number of filters in first layer of discriminator
batch_size = 1 # batch_size
pool_size = 50 # pool_size  Generated Image Pool
img_width = 256 # Imput image will of width 256
img_height = 256 # Input image will be of height 256
img_depth = 3 # RGB format
```

### Encoding:

```python
def instance_norm(x):
    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale',[x.get_shape()[-1]], 
            initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable('offset',[x.get_shape()[-1]],
                                 initializer=tf.constant_initializer(0.0))
        out = scale*tf.div(x-mean, tf.sqrt(var+epsilon)) + offset
        return out
```

```python
def general_conv2d(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1):
    with tf.variable_scope(name):
        conv = tf.contrib.layers.conv2d(inputconv, num_features, [window_width, 
        			  window_height], [stride_width, stride_height],padding, 
        			  activation_fn=None, 
                      weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                      biases_initializer=tf.constant_initializer(0.0))
        if do_norm:
            conv = instance_norm(conv)            
        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv,"relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")
        return conv
```

```python
def build_resnet_block(inputres, dim, name="resnet"):    
    with tf.variable_scope(name):
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "VALID","c1")
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, 0.02, 
                                 "VALID","c2",do_relu=False)        
        return tf.nn.relu(out_res + inputres)
```

```python
def build_generator(input_gen):
    # input_gen.shape = (256, 256, 3)
    o_c1 = general_conv2d(input_gen, num_features=ngf, window_width=7, window_height=7, stride_width=1, stride_height=1)
     # o_c1.shape = (256, 256, 64)
    o_c2 = general_conv2d(o_c1, num_features=ngf*2, window_width=3, window_height=3, stride_width=2, stride_height=2)
    # o_c2.shape = (128, 128, 128)
    o_enc_A = general_conv2d(o_c2, num_features=ngf*4, window_width=3, window_height=3, stride_width=2, stride_height=2)
    # o_enc_A.shape = (64, 64, 256)

    # Transformation
    o_r1 = build_resnet_block(o_enc_A, num_features=64*4)
    o_r2 = build_resnet_block(o_r1, num_features=64*4)
    o_r3 = build_resnet_block(o_r2, num_features=64*4)
    o_r4 = build_resnet_block(o_r3, num_features=64*4)
    o_r5 = build_resnet_block(o_r4, num_features=64*4)
    o_enc_B = build_resnet_block(o_r5, num_features=64*4)
	# o_enc_B.shape = (64, 64, 256)
    
    #Decoding
    o_d1 = general_deconv2d(o_enc_B, num_features=ngf*2 window_width=3, window_height=3, stride_width=2, stride_height=2)
    o_d2 = general_deconv2d(o_d1, num_features=ngf, window_width=3, window_height=3, stride_width=2, stride_height=2)
    gen_B = general_conv2d(o_d2, num_features=3, window_width=7, window_height=7, stride_width=1, stride_height=1)
    #gen_B.shape = (256, 256, 3) 

    return gen_B
```

### Building the discriminator

We discussed how to build a generator, however for adversarial training of the network we need to build a discriminator as well. The discriminator would take an image as an input and try to predict if it is an original or the output from the generator. Generator can be visualized in following image.

![Model Architecture 1](https://hardikbansal.github.io/CycleGANBlog/images/discriminator.jpg)

The discriminator is simply a convolution network in our case. First, we will extract the features from the image.

```python
def build_gen_discriminator(inputdisc, name="discriminator"):
    with tf.variable_scope(name):
        f = 4
        o_c1 = general_conv2d(inputdisc, ndf, f, f, 2, 2, 0.02, 
                              "SAME", "c1", do_norm=False, relufactor=0.2)
        o_c2 = general_conv2d(o_c1, ndf*2, f, f, 2, 2, 0.02, 
                              "SAME", "c2", relufactor=0.2)
        o_c3 = general_conv2d(o_c2, ndf*4, f, f, 2, 2, 0.02, 
                              "SAME", "c3", relufactor=0.2)
        o_c4 = general_conv2d(o_c3, ndf*8, f, f, 1, 1, 0.02, 
                              "SAME", "c4",relufactor=0.2)
        o_c5 = general_conv2d(o_c4, 1, f, f, 1, 1, 0.02, 
                              "SAME", "c5",do_norm=False,do_relu=False)
        return o_c5
```

We now have two main components of the model, namely **Generator** and **Discriminator**, and since we want to make this model work in both the direction i.e., from A→B and from B→A, we will have two Generators, namely GeneratorA→B and GeneratorB→A, and two Discriminators, namely DiscriminatorA  and  DiscriminatorB.

### Building the model

Before getting to loss funtion let us define the base and see how to take input, construct the model.

```python
input_A = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_A")
input_B = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_B")
```

These placeholders will act as input while defining our model as follow.

```python
gen_B = build_generator(input_A, name="generator_AtoB")
gen_A = build_generator(input_B, name="generator_BtoA")
dec_A = build_discriminator(input_A, name="discriminator_A")
dec_B = build_discriminator(input_B, name="discriminator_B")

dec_gen_A = build_discriminator(gen_A, "discriminator_A")
dec_gen_B = build_discriminator(gen_B, "discriminator_B")
cyc_A = build_generator(gen_B, "generator_BtoA")
cyc_B = build_generator(gen_A, "generator_AtoB")
```

Above variable names are quite intuitive in nature. gengen represents image generated after using corresponding Generator and decdec represents decision after feeding the corresponding input to the discriminator.

### Loss Function

By now we have two generators and two discriminators. We need to design the loss function in a way which accomplishes our goal. The loss function can be seen having four parts:

1. Discriminator must approve all the original images of the corresponding categories.
2. Discriminator must reject all the images which are generated by corresponding Generators to fool them.
3. Generators must make the discriminators approve all the generated images, so as to fool them.
4. The generated image must retain the property of original image, so if we generate a fake image using a generator say GeneratorA→B then we must be able to get back to original image using the another generator GeneratorB→A- it must satisfy cyclic-consistency.

##### Discriminator loss

###### Part 1

Discriminator must be trained such that recommendation for images from category A must be as close to 1, and vice versa for discriminator B. So Discriminator A would like to minimize (DiscriminatorA(a)−1)2(DiscriminatorA(a)−1)2 and same goes for B as well. This can be implemented as:

```python
D_A_loss_1 = tf.reduce_mean(tf.squared_difference(dec_A,1))
D_B_loss_1 = tf.reduce_mean(tf.squared_difference(dec_B,1))
```

###### Part 2

Since, discriniator should be able to distinguish between generated and original images, it should also be predicting 0 for images produced by the generator, i.e. Discriminator A wwould like to minimize (DiscriminatorA(GeneratorB→A(b)))2(DiscriminatorA(GeneratorB→A(b)))2. It can be calculated as follow:

```python
D_A_loss_2 = tf.reduce_mean(tf.square(dec_gen_A))
D_B_loss_2 = tf.reduce_mean(tf.square(dec_gen_B))

D_A_loss = (D_A_loss_1 + D_A_loss_2)/2
D_B_loss = (D_B_loss_1 + D_B_loss_2)/2
```

##### Generator loss

Generator should eventually be able to fool the discriminator about the authencity of it's generated images. This can done if the recommendation by discriminator for the generated images is as close to 1 as possible. So generator would like to minimize (DiscriminatorB(GeneratorA→B(a))−1)2(DiscriminatorB(GeneratorA→B(a))−1)2 So the loss is:

```python
g_loss_B_1 = tf.reduce_mean(tf.squared_difference(dec_gen_A,1))
g_loss_A_1 = tf.reduce_mean(tf.squared_difference(dec_gen_A,1))
```

##### Cyclic loss

And the last one and one of the most important one is the cyclic loss that captures that we are able to get the image back using another generator and thus the difference between the original image and the cyclic image should be as small as possible.

```python
cyc_loss = tf.reduce_mean(tf.abs(input_A-cyc_A)) + tf.reduce_mean(tf.abs(input_B-cyc_B))
```

The complete generator loss is then:

```python
g_loss_A = g_loss_A_1 + 10*cyc_loss
g_loss_B = g_loss_B_1 + 10*cyc_loss
```

The multiplicative factor of 10 for **cyc_loss** assigns more importance to cyclic loss than the discrimination loss.

##### Putting it together

With the loss function defined, all the is needed to train the model is to minimize the loss function w.r.t. model parameters.

```python
d_A_trainer = optimizer.minimize(d_loss_A, var_list=d_A_vars)
d_B_trainer = optimizer.minimize(d_loss_B, var_list=d_B_vars)
g_A_trainer = optimizer.minimize(g_loss_A, var_list=g_A_vars)
g_B_trainer = optimizer.minimize(g_loss_B, var_list=g_B_vars)
```

### Training the model

```python
for epoch in range(0,100):
    # Define the learning rate schedule. The learning rate is kept
    # constant upto 100 epochs and then slowly decayed
    if(epoch < 100) :
        curr_lr = 0.0002
    else:
        curr_lr = 0.0002 - 0.0002*(epoch-100)/100

    # Running the training loop for all batches
    for ptr in range(0,num_images):

        # Train generator G_A->B
        _, gen_B_temp = sess.run([g_A_trainer, gen_B],
                                 feed_dict={input_A:A_input[ptr], 
                                            input_B:B_input[ptr], lr:curr_lr})

        # We need gen_B_temp because to calculate the error in training D_B
        _ = sess.run([d_B_trainer],
                     feed_dict={input_A:A_input[ptr], 
                                input_B:B_input[ptr], 
                                lr:curr_lr})

        # Same for G_B->A and D_A as follow
        _, gen_A_temp = sess.run([g_B_trainer, gen_A], 
                                 feed_dict={input_A:A_input[ptr], 
                                            input_B:B_input[ptr], lr:curr_lr})
        _ = sess.run([d_A_trainer],
                     feed_dict={input_A:A_input[ptr], 
                                input_B:B_input[ptr], lr:curr_lr})
```

You can see in above training function that one by one we are calling trainers corresponding to different Dicriminators and Generators. For training them, we need to feed traing images and learning rate of the optimizer. Since, we have **batch_size** = 1, so, **num_batches = num_images**.

Since, we are nearly done with the code, below is look at the default parameters that we took to train the model

##### Generated Image Pool

Calculating the discriminator loss for each generated image would be computationally prohibitive. To speed up training we store a collection of previously generated images for each domain and to use only one of these images for calculating the error. First, fill the image_pool one by one until its full and after that randomly replace an image from the pool and store the latest one and use the replaced image for training in that iteration.

```python
def image_pool(self, num_gen, gen_img, gen_pool):
    if(num_gen < pool_size):
        gen_img_pool[num_gen] = gen_img
        return gen_img
    else :
        p = random.random()
        if p > 0.5:
            # Randomly selecting an id to return for calculating the discriminator loss
            random_id = random.randint(0,pool_size-1)
            temp = gen_img_pool[random_id]
            gen_pool[random_id] = gen_img
            return temp
        else :
            return gen_img
```

```python
gen_image_pool_A = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="gen_img_pool_A")
gen_image_pool_B = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="gen_img_pool_B")
gen_pool_rec_A = build_gen_discriminator(gen_image_pool_A, "d_A")
gen_pool_rec_B = build_gen_discriminator(gen_image_pool_B, "d_B")
# Also the discriminator loss will change as follow
D_A_loss_2 = tf.reduce_mean(tf.square(gen_pool_rec_A))
D_A_loss_2 = tf.reduce_mean(tf.square(gen_pool_rec_A))
```

The image pool requires minor modifications to the code. For complete code refer to the implementation [here](https://github.com/hardikbansal/CycleGAN/). 