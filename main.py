{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "8f951ed1-4273-48d9-81ed-ef7ac959b9ca",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\tensorflow\\python\\keras\\engine\\training_arrays_v1.py:37: UserWarning: A NumPy version >=1.22.4 and <1.29.0 is required for this version of SciPy (detected version 2.0.2)\n",
      "  from scipy.sparse import issparse  # pylint: disable=g-import-not-at-top\n",
      "\n",
      "A module that was compiled using NumPy 1.x cannot be run in\n",
      "NumPy 2.0.2 as it may crash. To support both 1.x and 2.x\n",
      "versions of NumPy, modules must be compiled with NumPy 2.0.\n",
      "Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.\n",
      "\n",
      "If you are a user of the module, the easiest solution will be to\n",
      "downgrade to 'numpy<2' or try to upgrade the affected module.\n",
      "We expect that some modules will need time to support NumPy 2.\n",
      "\n",
      "Traceback (most recent call last):  File \"<frozen runpy>\", line 198, in _run_module_as_main\n",
      "  File \"<frozen runpy>\", line 88, in _run_code\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\ipykernel_launcher.py\", line 17, in <module>\n",
      "    app.launch_new_instance()\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\traitlets\\config\\application.py\", line 1077, in launch_instance\n",
      "    app.start()\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\ipykernel\\kernelapp.py\", line 739, in start\n",
      "    self.io_loop.start()\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\tornado\\platform\\asyncio.py\", line 205, in start\n",
      "    self.asyncio_loop.run_forever()\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\asyncio\\base_events.py\", line 607, in run_forever\n",
      "    self._run_once()\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\asyncio\\base_events.py\", line 1922, in _run_once\n",
      "    handle._run()\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\asyncio\\events.py\", line 80, in _run\n",
      "    self._context.run(self._callback, *self._args)\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\ipykernel\\kernelbase.py\", line 529, in dispatch_queue\n",
      "    await self.process_one()\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\ipykernel\\kernelbase.py\", line 518, in process_one\n",
      "    await dispatch(*args)\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\ipykernel\\kernelbase.py\", line 424, in dispatch_shell\n",
      "    await result\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\ipykernel\\kernelbase.py\", line 766, in execute_request\n",
      "    reply_content = await reply_content\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\ipykernel\\ipkernel.py\", line 429, in do_execute\n",
      "    res = shell.run_cell(\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\ipykernel\\zmqshell.py\", line 549, in run_cell\n",
      "    return super().run_cell(*args, **kwargs)\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\IPython\\core\\interactiveshell.py\", line 3048, in run_cell\n",
      "    result = self._run_cell(\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\IPython\\core\\interactiveshell.py\", line 3103, in _run_cell\n",
      "    result = runner(coro)\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\IPython\\core\\async_helpers.py\", line 129, in _pseudo_sync_runner\n",
      "    coro.send(None)\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\IPython\\core\\interactiveshell.py\", line 3308, in run_cell_async\n",
      "    has_raised = await self.run_ast_nodes(code_ast.body, cell_name,\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\IPython\\core\\interactiveshell.py\", line 3490, in run_ast_nodes\n",
      "    if await self.run_code(code, result, async_=asy):\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\IPython\\core\\interactiveshell.py\", line 3550, in run_code\n",
      "    exec(code_obj, self.user_global_ns, self.user_ns)\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Temp\\ipykernel_7864\\1399570729.py\", line 3, in <module>\n",
      "    import tensorflow as tf\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\tensorflow\\__init__.py\", line 49, in <module>\n",
      "    from tensorflow._api.v2 import __internal__\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\tensorflow\\_api\\v2\\__internal__\\__init__.py\", line 13, in <module>\n",
      "    from tensorflow._api.v2.__internal__ import feature_column\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\tensorflow\\_api\\v2\\__internal__\\feature_column\\__init__.py\", line 8, in <module>\n",
      "    from tensorflow.python.feature_column.feature_column_v2 import DenseColumn # line: 1777\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\tensorflow\\python\\feature_column\\feature_column_v2.py\", line 38, in <module>\n",
      "    from tensorflow.python.feature_column import feature_column as fc_old\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\tensorflow\\python\\feature_column\\feature_column.py\", line 41, in <module>\n",
      "    from tensorflow.python.layers import base\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\tensorflow\\python\\layers\\base.py\", line 16, in <module>\n",
      "    from tensorflow.python.keras.legacy_tf_layers import base\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\tensorflow\\python\\keras\\__init__.py\", line 25, in <module>\n",
      "    from tensorflow.python.keras import models\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\tensorflow\\python\\keras\\models.py\", line 25, in <module>\n",
      "    from tensorflow.python.keras.engine import training_v1\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\tensorflow\\python\\keras\\engine\\training_v1.py\", line 46, in <module>\n",
      "    from tensorflow.python.keras.engine import training_arrays_v1\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\tensorflow\\python\\keras\\engine\\training_arrays_v1.py\", line 37, in <module>\n",
      "    from scipy.sparse import issparse  # pylint: disable=g-import-not-at-top\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\scipy\\sparse\\__init__.py\", line 295, in <module>\n",
      "    from ._csr import *\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\scipy\\sparse\\_csr.py\", line 11, in <module>\n",
      "    from ._sparsetools import (csr_tocsc, csr_tobsr, csr_count_blocks,\n"
     ]
    },
    {
     "ename": "AttributeError",
     "evalue": "_ARRAY_API not found",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "\u001b[1;31mAttributeError\u001b[0m: _ARRAY_API not found"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      "A module that was compiled using NumPy 1.x cannot be run in\n",
      "NumPy 2.0.2 as it may crash. To support both 1.x and 2.x\n",
      "versions of NumPy, modules must be compiled with NumPy 2.0.\n",
      "Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.\n",
      "\n",
      "If you are a user of the module, the easiest solution will be to\n",
      "downgrade to 'numpy<2' or try to upgrade the affected module.\n",
      "We expect that some modules will need time to support NumPy 2.\n",
      "\n",
      "Traceback (most recent call last):  File \"<frozen runpy>\", line 198, in _run_module_as_main\n",
      "  File \"<frozen runpy>\", line 88, in _run_code\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\ipykernel_launcher.py\", line 17, in <module>\n",
      "    app.launch_new_instance()\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\traitlets\\config\\application.py\", line 1077, in launch_instance\n",
      "    app.start()\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\ipykernel\\kernelapp.py\", line 739, in start\n",
      "    self.io_loop.start()\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\tornado\\platform\\asyncio.py\", line 205, in start\n",
      "    self.asyncio_loop.run_forever()\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\asyncio\\base_events.py\", line 607, in run_forever\n",
      "    self._run_once()\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\asyncio\\base_events.py\", line 1922, in _run_once\n",
      "    handle._run()\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\asyncio\\events.py\", line 80, in _run\n",
      "    self._context.run(self._callback, *self._args)\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\ipykernel\\kernelbase.py\", line 529, in dispatch_queue\n",
      "    await self.process_one()\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\ipykernel\\kernelbase.py\", line 518, in process_one\n",
      "    await dispatch(*args)\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\ipykernel\\kernelbase.py\", line 424, in dispatch_shell\n",
      "    await result\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\ipykernel\\kernelbase.py\", line 766, in execute_request\n",
      "    reply_content = await reply_content\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\ipykernel\\ipkernel.py\", line 429, in do_execute\n",
      "    res = shell.run_cell(\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\ipykernel\\zmqshell.py\", line 549, in run_cell\n",
      "    return super().run_cell(*args, **kwargs)\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\IPython\\core\\interactiveshell.py\", line 3048, in run_cell\n",
      "    result = self._run_cell(\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\IPython\\core\\interactiveshell.py\", line 3103, in _run_cell\n",
      "    result = runner(coro)\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\IPython\\core\\async_helpers.py\", line 129, in _pseudo_sync_runner\n",
      "    coro.send(None)\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\IPython\\core\\interactiveshell.py\", line 3308, in run_cell_async\n",
      "    has_raised = await self.run_ast_nodes(code_ast.body, cell_name,\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\IPython\\core\\interactiveshell.py\", line 3490, in run_ast_nodes\n",
      "    if await self.run_code(code, result, async_=asy):\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\IPython\\core\\interactiveshell.py\", line 3550, in run_code\n",
      "    exec(code_obj, self.user_global_ns, self.user_ns)\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Temp\\ipykernel_7864\\1399570729.py\", line 3, in <module>\n",
      "    import tensorflow as tf\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\tensorflow\\__init__.py\", line 49, in <module>\n",
      "    from tensorflow._api.v2 import __internal__\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\tensorflow\\_api\\v2\\__internal__\\__init__.py\", line 13, in <module>\n",
      "    from tensorflow._api.v2.__internal__ import feature_column\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\tensorflow\\_api\\v2\\__internal__\\feature_column\\__init__.py\", line 8, in <module>\n",
      "    from tensorflow.python.feature_column.feature_column_v2 import DenseColumn # line: 1777\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\tensorflow\\python\\feature_column\\feature_column_v2.py\", line 38, in <module>\n",
      "    from tensorflow.python.feature_column import feature_column as fc_old\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\tensorflow\\python\\feature_column\\feature_column.py\", line 41, in <module>\n",
      "    from tensorflow.python.layers import base\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\tensorflow\\python\\layers\\base.py\", line 16, in <module>\n",
      "    from tensorflow.python.keras.legacy_tf_layers import base\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\tensorflow\\python\\keras\\__init__.py\", line 25, in <module>\n",
      "    from tensorflow.python.keras import models\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\tensorflow\\python\\keras\\models.py\", line 25, in <module>\n",
      "    from tensorflow.python.keras.engine import training_v1\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\tensorflow\\python\\keras\\engine\\training_v1.py\", line 71, in <module>\n",
      "    from scipy.sparse import issparse  # pylint: disable=g-import-not-at-top\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\scipy\\sparse\\__init__.py\", line 295, in <module>\n",
      "    from ._csr import *\n",
      "  File \"C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\scipy\\sparse\\_csr.py\", line 11, in <module>\n",
      "    from ._sparsetools import (csr_tocsc, csr_tobsr, csr_count_blocks,\n"
     ]
    },
    {
     "ename": "AttributeError",
     "evalue": "_ARRAY_API not found",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
      "\u001b[1;31mAttributeError\u001b[0m: _ARRAY_API not found"
     ]
    }
   ],
   "source": [
    "# Step 1: Import Libraries and Load the Model\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.datasets import imdb\n",
    "from tensorflow.keras.preprocessing import sequence\n",
    "from tensorflow.keras.models import load_model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "cb591776-0110-4dd3-b8c0-122e87f400b8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the IMDB dataset word index\n",
    "word_index = imdb.get_word_index()\n",
    "reverse_word_index = {value: key for key, value in word_index.items()}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "4a5eb0b1-96ec-468d-b641-dda04b80447a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 2: Helper Functions\n",
    "# Function to decode reviews\n",
    "def decode_review(encoded_review):\n",
    "    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])\n",
    "\n",
    "# Function to preprocess user input\n",
    "def preprocess_text(text):\n",
    "    words = text.lower().split()\n",
    "    encoded_review = [word_index.get(word, 2) + 3 for word in words]\n",
    "    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)\n",
    "    return padded_review"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "42223ffd-33d4-47ee-8007-50abaf1d2ff7",
   "metadata": {},
   "outputs": [],
   "source": [
    "### Prediction  function\n",
    "\n",
    "def predict_sentiment(review):\n",
    "    preprocessed_input=preprocess_text(review)\n",
    "\n",
    "    prediction=model.predict(preprocessed_input)\n",
    "\n",
    "    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'\n",
    "    \n",
    "    return sentiment, prediction[0][0]\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "255c56ea-77c5-4626-9071-b93d8804211a",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "## streamlit app\n",
    "# Streamlit app\n",
    "st.title('IMDB Movie Review Sentiment Analysis')\n",
    "st.write('Enter a movie review to classify it as positive or negative.')\n",
    "\n",
    "# User input\n",
    "user_input = st.text_area('Movie Review')\n",
    "\n",
    "if st.button('Classify'):\n",
    "\n",
    "    preprocessed_input=preprocess_text(user_input)\n",
    "\n",
    "    ## MAke prediction\n",
    "    prediction=model.predict(preprocessed_input)\n",
    "    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'\n",
    "\n",
    "    # Display the result\n",
    "    st.write(f'Sentiment: {sentiment}')\n",
    "    st.write(f'Prediction Score: {prediction[0][0]}')\n",
    "else:\n",
    "    st.write('Please enter a movie review.')\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ecbdc2e7-dba1-482f-bf4a-61bdb8cd6bfd",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  },
  "widgets": {
   "application/vnd.jupyter.widget-state+json": {
    "state": {},
    "version_major": 2,
    "version_minor": 0
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
