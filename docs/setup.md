1. git clone <your-repo-url>
2. cd ML-API-Project
3. python -m venv venv && source venv/bin/activate

For Dev 
4. pip install -r requirements.dev.txt
Else
5. pip install -r requirements.txt
6. uvicorn main:app --reload


Setup Docker Container
1. docker build -t ${image_name} .
2. docker run -d --name ${container_name} -p 8000:8000 ${image_name}
3. docker ps
4. docker images
5. docker exec -it ${running_container_id} sh
5. docker logs -f ${running_container_id}