# Optional nginx reverse proxy with basic auth.

# Generate password file:
#   docker run --rm httpd:2.4-alpine htpasswd -nbB admin 'your-password' > nginx/.htpasswd
#
# Start:
#   docker compose --profile with-nginx up --build
#
# App via nginx: http://localhost:8080
# Direct Streamlit: http://localhost:8501
