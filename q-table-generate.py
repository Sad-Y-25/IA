import xml.etree.ElementTree as ET
import os

add_xml_path = 'data/networks/RL.add.xml'
det_output_folder = 'det_outputs'

# Create folder
if not os.path.exists(det_output_folder):
    os.makedirs(det_output_folder)

# Parse XML
tree = ET.parse(add_xml_path)
root = tree.getroot()

# Update each detector's file attribute
for detector in root.findall('laneAreaDetector'):
    old_file = detector.get('file')
    if old_file:
        new_file = os.path.join(det_output_folder, os.path.basename(old_file))
        detector.set('file', new_file)

# Save
tree.write(add_xml_path)
print('Updated RL.add.xml to use det_outputs/ for e*.xml')