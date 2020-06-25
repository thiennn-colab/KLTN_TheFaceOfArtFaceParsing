from django.db import models
from django.conf import settings

# Create your models here.


def upload_update_image(instance, filename):
    return "updates/{user}/{filename}".format(user=instance.user, filename=filename)


class Update(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.DO_NOTHING)
    content = models.TextField(blank=True, null=True)
    image = models.ImageField(
        upload_to=upload_update_image, blank=True, null=True)

    def __str__(self):
        return self.content
