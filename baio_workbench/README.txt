#here we build a docker image with all dependancies for baio
#to update the baio code within the image:
    #replace the current 'baio' dir with the new one

#add any dependancies to the requirements.txt 
#execute the command below by replacing the TAG with your version

docker build -t baio:<TAG> .

#push the command on docker-hub or distribute it however


#to make run the image and make it automatically launch in your default browser:

nano baio

#input this in your script by adapting potential ports if you want
#!/bin/bash

# Use the full path to Docker if necessary
/usr/local/bin/docker run -p 8501:8501 baio:latest &
sleep 3
open http://localhost:8501

docker rmi baio:0.0.4 noahbruderer/baio:0.0.4
