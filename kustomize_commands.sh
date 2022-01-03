kustomize build k8s/base | kubectl apply -f -
kustomize build k8s/base | kubectl delete -f -