import tensorflow as tf

# Enable logging
tf.logging.set_verbosity(tf.logging.INFO)

def conv_net(x_dict, n_classes, img_size, dropout, reuse, is_training):
    
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['images']

        # The data input is a 1-D vector of 2304 features (48*48 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # 4D Input-Tensor: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, img_size, img_size, 1])

        # Write images to summary for TensorBoard
        tf.summary.image("img", x)

        # Convolution Layers with 32 filters and a kernel size of 3.
        conv1 = tf.layers.conv2d(x, 32, 3, padding='same', activation=tf.nn.relu)
        conv1 = tf.layers.conv2d(conv1, 32, 3, padding='same', activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layers with 64 filters and a kernel size of 3.
        conv2 = tf.layers.conv2d(conv1, 64, 3, padding='same', activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv2, 64, 3, padding='same', activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2.
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        
        # Convolution Layer with 128 filters and a kernel size of 3
        conv3 = tf.layers.conv2d(conv2, 128, 3, padding='same', activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(conv3, 128, 3, padding='same', activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(conv3, 128, 3, padding='same', activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2.
        conv3 = tf.layers.max_pooling2d(conv3, 2, 2)

        # Convolution Layers with 256 filters and a kernel size of 3.
        conv4 = tf.layers.conv2d(conv3, 256, 3, padding='same', activation=tf.nn.relu)
        conv4 = tf.layers.conv2d(conv4, 256, 3, padding='same', activation=tf.nn.relu)
        conv4 = tf.layers.conv2d(conv4, 256, 3, padding='same', activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2.
        conv4 = tf.layers.max_pooling2d(conv4, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer.
        fc = tf.contrib.layers.flatten(conv4)

        # Fully connected layer.
        fc = tf.layers.dense(fc, 1024)
        # Apply Dropout (only when is_training is True).
        fc = tf.layers.dropout(fc, rate=dropout, training=is_training)

        # Output layer, class prediction.
        logits = tf.layers.dense(fc, n_classes)

    return logits

def model_fn(features, labels, mode, params):

    # Because Dropout has different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, params['num_classes'], params['img_size'], params['dropout_rate'], reuse=False, is_training=True)
    logits_test = conv_net(features, params['num_classes'], params['img_size'], params['dropout_rate'], reuse=True, is_training=False)
    
    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode, 
            predictions=pred_classes,
            export_outputs={'classes': tf.estimator.export.PredictOutput(pred_classes)})
    else:
        # Define loss across all classes
        loss_op = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits_train,
                labels=tf.nn.softmax(
                    tf.cast(labels, dtype=tf.float32)
                )
            )
        )

        # Define Optimizer with learning rate decay
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])

        # Define Train Operation
        train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
        
        # Accuracy for TensorBoard summary
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(labels, axis=1), tf.int64), pred_classes), tf.float32))
        # Accuracy metric for train_and_eval function
        acc_op = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=pred_classes)

        # Write summaries for TensorBoard
        tf.summary.scalar("loss", loss_op)
        tf.summary.scalar("accuracy", acc)
        
        # The model function needs to return an EstimatorSpec
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_classes,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'accuracy': acc_op})

    return spec