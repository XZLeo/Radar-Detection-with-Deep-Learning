FROM zocker.cicd.autoheim.net/mw/madame_web:20220324_200124_aa12ee255

RUN pip install radar-scenes \
    imgaug \
    terminaltables \
    torchsummary

