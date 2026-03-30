<h1 align="center">SEAMLESSLY NATURAL</h1>

<p align="center">
  <img src="first.JPG" alt="SEAMLESSLY NATURAL" width="1000">
</p>
<p align="center">
  <em>An image stitching algorithm for high-quality panoramic reconstruction</em>
</p>

<hr>

<h2>🔍 Description</h2>

<p>
  This repository presents <strong>SEAMLESSLY NATURAL</strong>, an image stitching algorithm designed to produce 
  <strong>high-quality panoramic images</strong>. The method preserves both <strong>global consistency</strong> 
  and <strong>local structures</strong>, while ensuring a <strong>visually natural appearance</strong>.
</p>

<h2>🧪 Dataset: UDIS-D</h2>

<p>
  Experiments were conducted using the <strong>UDIS-D</strong> dataset. Since the proposed algorithm relies on 
  the input image order, the dataset was reorganized into <strong>ordered sequences</strong>. The resulting version 
  used in our experiments is referred to as <strong>UDIS_datasetuni</strong>.
</p>

<h2>📊 Evaluation: BRISQUE (BRI) Scores</h2>

<p>
  Image quality is evaluated using <strong>BRISQUE</strong>, a no-reference perceptual quality metric 
  that assesses the naturalness of the generated images.
</p>

<h3>📂 Evaluation Procedure</h3>

<p>This metric is computed on the final stitched images produced by the method.</p>

<ol>
  <li>
    Use the reorganized dataset <code>UDIS_datasetuni</code>, where images are arranged in the correct order.
  </li>
  <li>
    To compute the BRISQUE (BRI) scores, run the evaluation script 
    <code>brisque_evaluation.py</code>.
  </li>
  <li>
    Example results are available in the folder 
    <code>"our results on UDIS dataset"</code>. However, the script can be used to evaluate 
    any set of stitched images.
  </li>
</ol>

<p>
  You can download the test dataset and example results obtained on different datasets here:<br>
  👉 https://drive.google.com/drive/folders/1CUV0PjbWwC7lh_VVOazLt5XnPYELgjfc?usp=drive_link
</p>

<hr>

<h2>⚙️ Requirements</h2>

<p>The project requires the following dependencies:</p>

<pre><code>
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scipy>=1.10.0
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0
Pillow>=10.0.0
imageio>=2.31.0
scikit-image>=0.21.0
matplotlib>=3.7.0
</code></pre>

<hr>

<h2>🚀 Usage</h2>

<p>Two execution options are provided:</p>

<ul>
  <li>
    <strong>Jupyter Notebook (.ipynb):</strong>  
    Can be executed using <strong>Google Colab</strong> or any local Jupyter environment.  
    This option is recommended for quick testing and visualization.
  </li>

  <li>
    <strong>Python Script (.py):</strong>  
    Can be executed locally in a standard Python environment for full control and integration into pipelines.
  </li>
</ul>

<p>
  Choose the option that best fits your workflow and computational setup.
</p>

<hr>