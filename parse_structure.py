
data="""write python script
read in lines of outline, has : separating filename from content
split out filename, number of stars and rest of text
if filename changes break
look at structure, if goes in up (less stars), break
if goes down more stars than before, capture pairs
"""
last_filename = ""
last_stars    = 0
last_content  = "start"

lines = []
with open("outline.txt") as fi:
    for l in fi:
        l = l.strip()
        parts = l.split(":")
        filename=parts[0]
        rest = ":".join(parts[1:])
        
        parts = rest.split(" ")
        stars=parts[0]
        #lines.append(l)


        content = " ".join(parts[1:])

        changed= 0
        if filename != last_filename :
            changed = 0
        elif len(stars) < last_stars :
            changed = 1
        if changed:
            print("|".join([str(last_stars), str(len(stars)),  last_content, content]))

        last_filename = filename
        last_stars    = len(stars)
        last_content  = content
        
# file_name_pairs = []
# current_filename = None
# current_number_of_stars = 0
# for line in lines:
#     if current_filename is None or line != current_filename:
#         # check for filename, split by ':' character and store filename and content separately
#         filename_content = line.split(':')
#         if len(filename_content) == 2:
#             current_filename = filename_content[0].strip()
#             if current_number_of_stars > int(filename_content[1].strip()):
#                 file_name_pairs.append((current_filename, current_number_of_stars))
#         current_number_of_stars = None
#     elif line.startswith('*') and current_number_of_stars is not None:
#         # check if number of stars changes, break loop if it does
#         if int(line.strip()[1:]) > current_number_of_stars:
#             break
#     else:
#         current_number_of_stars = None
# if len(file_name_pairs) == 0:
#     print("No file names found in the outline.")
# else:
#     print(file_name_pairs)
