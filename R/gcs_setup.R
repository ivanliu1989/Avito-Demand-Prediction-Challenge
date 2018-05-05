library(googleComputeEngineR)

project = 'avito-203110'
zone = 'australia-southeast1-b'
account_key = "D:/Projects/Avito-Demand-Prediction-Challenge/gcs-key-avito.json"
# account_key = "./gcs-key-avito.json"

Sys.setenv(GCE_AUTH_FILE = account_key,
           GCE_DEFAULT_PROJECT_ID = project,
           GCE_DEFAULT_ZONE = zone)
Sys.getenv("GCE_AUTH_FILE")

options(googleAuthR.scopes.selected = "https://www.googleapis.com/auth/cloud-platform")
gce_auth()

gce_get_project()

gce_global_project(project)
gce_global_zone(zone)
default_project = gce_get_project(project)
default_project$name

vm = gce_vm(template = 'rstudio',
            name = 'rstudio-avito',
            username = 'rstudio',
            password = 'Password2018',
            predefined_type = 'n1-highmem-2')

my_rstudio = gce_vm('rstudio-avito')


job = gce_vm_stop('rstudio-avito')

gce_list_instances()
