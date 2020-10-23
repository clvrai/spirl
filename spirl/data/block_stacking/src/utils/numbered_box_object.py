import os
import xml.etree.ElementTree as ET
from spirl.data.block_stacking.src.robosuite.models.objects import BoxObject


class NumberedBoxObject(BoxObject):
    def __init__(self, **kwargs):
        self.number = kwargs.pop("number")
        super().__init__(**kwargs)

        if self.number is not None:
            asset_dir = os.path.join(os.getcwd(), "spirl/data/block_stacking/assets")
            texture_path = os.path.join(asset_dir, "textures/obj{}.png".format(self.number))

            texture = ET.SubElement(self.asset, "texture")
            texture.set("file", texture_path)
            texture.set("name", "obj-{}-texture".format(self.number))
            texture.set("gridsize", "1 2")
            texture.set("gridlayout", ".U")
            texture.set("rgb1", "1.0 1.0 1.0")
            texture.set("vflip", "true")
            texture.set("hflip", "true")

            material = ET.SubElement(self.asset, "material")
            material.set("name", "obj-{}-material".format(self.number))
            material.set("reflectance", "0.5")
            material.set("specular", "0.5")
            material.set("shininess", "0.1")
            material.set("texture", "obj-{}-texture".format(self.number))
            material.set("texuniform", "false")

    def get_collision_attrib_template(self):
        template = super().get_collision_attrib_template()
        if self.number is not None:
            template.update({"material": "obj-{}-material".format(self.number)})
        return template

    def get_visual_attrib_template(self):
        template = super().get_visual_attrib_template()
        if self.number is not None:
            template.update({"material": "obj-{}-material".format(self.number)})
        return template
