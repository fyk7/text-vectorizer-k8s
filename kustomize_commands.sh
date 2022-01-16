kustomize build k8s/local | kubectl apply -f -
kustomize build k8s/local | kubectl delete -f -

kustomize build k8s/overlays/stg | kubectl apply -f -
kustomize build k8s/overlays/stg | kubectl delete -f -