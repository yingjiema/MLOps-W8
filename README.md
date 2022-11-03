<p align = "center" draggable=”false”
   ><img src="https://user-images.githubusercontent.com/37101144/161836199-fdb0219d-0361-4988-bf26-48b0fad160a3.png"
     width="200px"
     height="auto"/>
</p>

# <h1 align="center" id="heading">Week 8 - Deploying Pet-Bokeh using Kubernetes with Nvidia Triton on EC2</h1>

## 📚 Learning Objectives

By the end of this session, you will be able to:

- Create and configure a minikube cluster
- Integrate minikube with Nvidia Trition Server
- Deploy a minikube cluster on EC2

## 📦 Deliverables

- A screenshot of kubectl dashboard
- A screenshot of your deployment


## Deployment on EC2

### Create EC2 Instance

- Go to EC2 console: <https://us-east-1.console.aws.amazon.com/ec2/home?region=us-east-1>
- Create EC2 instance
- Pick deep learning ami
- Pick instance type: At least p3.2xlarge
- Create key-pair
- Download key
- Edit network
- Enable IPV4 address
- Open ports 8000-8004 from anywhere
- Launch Instance

### Install dependencies

- Get the ip address of the instance
- Change key permissions to 400 (`chmod 400 key.pem`)
- SSH into the machine `ssh -i key.pem ec2-user@ec2.ip.address`
- Install git if needed (`sudo apt install git` for ubuntu based distros, `sudo yum install git` for amazon linux)
- Install Docker (`sudo apt install docker` for ubuntu based distros, `sudo yum install docker` for amazon linux)
- Start Docker (`sudo systemctl start docker`)
- Add user to docker group (`sudo usermod -aG docker ${USER}`)
- Logout and Login again through SSH to take the group changes into account
- Check if docker installed correctly (`docker run hello-world`)
- Install minikube

```
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

```

- Start a minikube cluster
- Set an alias for kubectl
- For triton we need to add the credentials as secrets

```
kubectl create secret generic aws-env --from-literal='AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY' --from-literal='AWS_SECRET_ACCESS_KEY=YOUR_SECRET_ACCESS_KEY' --from-literal='AWS_DEFAULT_REGION=us-east-1'
```

### Locally built images

- Use minikube's docker
- Pull the triton image
- Build all the docker images

```
docker build -t main .
docker build -t face-emotion face-emotion/
docker build -t pet-bokeh pet-bokeh/
```

- Load all the kubernetes resources
- Forward the main port (`kubectl port-forward svc/main 8004:8004 --address 0.0.0.0`)

### Pulling from ECR

- Configure credentials
- Enable the addon
- Edit K8s/ECR/*.yaml to use your ECRs
- Load all the kubernetes resources
- Forward the main port (`kubectl port-forward svc/main 8004:8004 --address 0.0.0.0`)
