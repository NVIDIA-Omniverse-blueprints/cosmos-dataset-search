{{/*
Standard labels for all OpenShift resources.
*/}}
{{- define "visual-search.labels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{- end }}

{{/*
NGC image pull secret name.
*/}}
{{- define "visual-search.ngcPullSecret.name" -}}
ngc-secret
{{- end }}

{{/*
NGC image pull secret — reusable template for dockerconfigjson secrets.
Usage: {{ include "visual-search.ngcPullSecret" (list $ "secret-name") }}
*/}}
{{- define "visual-search.ngcPullSecret" -}}
apiVersion: v1
kind: Secret
metadata:
  name: {{ index . 1 }}
  namespace: {{ (index . 0).Release.Namespace }}
  labels:
    {{- include "visual-search.labels" (index . 0) | nindent 4 }}
type: kubernetes.io/dockerconfigjson
stringData:
  .dockerconfigjson: |
    {
      "auths": {
        "nvcr.io": {
          "username": "$oauthtoken",
          "password": {{ (index . 0).Values.openshift.secrets.ngcApiKey | quote }}
        }
      }
    }
{{- end }}
