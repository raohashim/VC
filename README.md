# Video-Coding
TU Ilmenau-Seminars Master Course Video Coding

To obtain the service state and relevant information, such as the scoring URI, Swagger URI, and authentication keys, you can use the following code:

primary, secondary = service.get_keys()

print('Service state:', service.state)
print('Service scoring URI:', service.scoring_uri)
print('Service Swagger URI:', service.swagger_uri)
print('Service primary authentication key:', primary)

