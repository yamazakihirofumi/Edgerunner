#scp -r ./master.py  dourlin@192.168.1.70:/home/dourlin/Develop/Edgerunner

current_time=$(date +"%Y-%m-%d %H:%M:%S")
echo $current_time
machine_name=$(hostname)
git commit -m "QC:$current_time on $machine_name" 
git push

