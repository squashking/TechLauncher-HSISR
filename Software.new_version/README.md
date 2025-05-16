1. This is a new version of GUI and the old will be replaced.
2. The Functions are copied from the old version. Will merge them with the GUI code in a more well orgainsed way.
3. Please manually download the trained model for Super-Resolution to the local directory:
<pre><code>cd Software.new_version
mkdir -p model/trained_model
cd model/trained_model
wget https://github.com/squashking/TechLauncher-HSISR/raw/refs/heads/main/Software/GUI/weights/fin_msdformer.pth
</code></pre>
4. To run the software:
<pre><code>cd Software.new_version
# source .venv/bin/activate  # activate the environment with the required packages
python3 main.py
</code></pre>
