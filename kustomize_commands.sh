kustomize build k8s/overlays/local | kubectl apply -f -
kustomize build k8s/overlays/local | kubectl delete -f -

kustomize build k8s/overlays/stg | kubectl apply -f -
kustomize build k8s/overlays/stg | kubectl delete -f -

kustomize build k8s/overlays/prd | kubectl apply -f -
kustomize build k8s/overlays/prd | kubectl delete -f -