import tensorflow as tf
import tensorflow_hub as hub
import tensorflowjs as tfjs
import os

# -----------------------------
# Step 1: Download TF-Hub model
# -----------------------------
print("Downloading TF-Hub model...")
hub_model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")
print("Model downloaded!")

# -----------------------------
# Step 2: Save original SavedModel (optional)
# -----------------------------
saved_model_dir = "saved_model"
if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)
print(f"Saving original SavedModel to '{saved_model_dir}' ...")
tf.saved_model.save(hub_model, saved_model_dir)

# -----------------------------
# Step 3: Check available signatures
# -----------------------------
loaded_model = tf.saved_model.load(saved_model_dir)
print("Available signatures in original model:", list(loaded_model.signatures.keys()))

# -----------------------------
# Step 4: Create wrapper for TF.js
# -----------------------------
class TFHubWrapper(tf.Module):
    def __init__(self, hub_model):
        super().__init__()
        self.hub_model = hub_model

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32, name="content_image"),
        tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32, name="style_image")
    ])
    def serve(self, content_image, style_image):
        # hub_model 返回 [tensor], 我们只取第0个
        stylized_image_list = self.hub_model(content_image, style_image)
        stylized_image = stylized_image_list[0]  # 取 tensor
        return {"output": stylized_image}

# -----------------------------
# Step 5: Save wrapped model with 'serving_default' signature
# -----------------------------
wrapper = TFHubWrapper(hub_model)
tfjs_saved_model_dir = "tfjs_saved_model_for_node"
if not os.path.exists(tfjs_saved_model_dir):
    os.makedirs(tfjs_saved_model_dir)

print(f"Saving TF.js compatible SavedModel to '{tfjs_saved_model_dir}' ...")
tf.saved_model.save(wrapper, tfjs_saved_model_dir, signatures={"serving_default": wrapper.serve})

# -----------------------------
# Step 6: Convert to TensorFlow.js format
# -----------------------------
tfjs_target_dir = "tfjs_model"
if not os.path.exists(tfjs_target_dir):
    os.makedirs(tfjs_target_dir)

print(f"Converting SavedModel to TensorFlow.js format in '{tfjs_target_dir}' ...")
tfjs.converters.convert_tf_saved_model(tfjs_saved_model_dir, tfjs_target_dir)

print("✅ Model conversion completed! Check the folder 'tfjs_model'.")
