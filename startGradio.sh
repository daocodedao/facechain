
workdir="$PWD"
cd $workdir

venvBinDir=venv/bin/
pythonPath=${workdir}/${venvBinDir}python
echo "Python path:  $pythonPath"

${pythonPath} app.py
