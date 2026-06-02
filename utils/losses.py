import tensorflow as tf
import config as cfg

def focal_loss(gamma=2.0, alpha=0.25, label_smoothing=0.1):
    def loss_fn(y_true, y_pred):
        # Convert true targets to one-hot encoding
        y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), depth=len(cfg.EMOTION_CLASSES))

        # Apply label smoothing
        n_classes = len(cfg.EMOTION_CLASSES)
        y_true_smoothed = y_true_oh * (1.0 - label_smoothing) + (label_smoothing / tf.cast(n_classes, tf.float32))

        # Clip predictions to avoid mathematical infinity log(0) errors
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Calculate cross-entropy
        cross_entropy = -y_true_smoothed * tf.math.log(y_pred)

        # Calculate probability of the true class
        p_t = tf.reduce_sum(y_true_oh * y_pred, axis=-1)
        p_t = tf.expand_dims(p_t, axis=-1) 

        # Calculate focal loss weighting factor
        focal_weight = alpha * tf.pow(1.0 - p_t, gamma)

        # Apply focal weight matrix to cross-entropy
        loss = tf.reduce_sum(focal_weight * cross_entropy, axis=-1)
        return loss
    return loss_fn