
--Run the below to build
gcloud builds submit --tag gcr.io/carboninequality/carbon-inequality-analysis-dashboard  --project=carboninequality

--Run the below to deploy to google cloud
gcloud run deploy --image gcr.io/carboninequality/carbon-inequality-analysis-dashboard --platform managed  --project=carboninequality --allow-unauthenticated
