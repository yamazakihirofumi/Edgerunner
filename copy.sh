#scp -r ./master.py  dourlin@latte:/home/dourlin/Develop/Edgerunner

current_time=$(date +"%Y-%m-%d %H:%M:%S")
echo $current_time
machine_name=$(hostname)
git commit -m "QC:$current_time on $machine_name" 
git push

