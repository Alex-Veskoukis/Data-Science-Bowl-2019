# Data-Science-Bowl-2019

* The file train_labels.csv has been provided to show how these groups would be computed on the assessments in the training set.   
* Assessment attempts are captured in event_code 4100 for all assessments except for Bird Measurer, which uses event_code 4110.    
* If the attempt was correct, it contains "correct":true.

* <strong>The intent of the competition is to use the gameplay data to forecast how many attempts a child will take to pass a given assessment (an incorrect answer is counted as an attempt).</strong>
* Each application install is represented by an installation_id. This will typically correspond to one child, but you should expect noise from issues such as shared devices.   
* In the training set, you are provided the full history of gameplay data.   
* In the test set, we have truncated the history after the start event of a single assessment, chosen randomly, for which you must predict the number of attempts.   
* Note that the training set contains many installation_ids which never took assessments, whereas every installation_id in the test set made an attempt on at least one assessment.    
      
      

## Train Data description
event_id - Randomly generated unique identifier for the event type. Maps to event_id column in specs table.    

game_session - Randomly generated unique identifier grouping events within a single game or video play session.   

timestamp - Client-generated datetime    

event_data - Semi-structured JSON formatted string containing the events parameters. Default fields are: event_count, event_code, and game_time; otherwise fields are determined by the event type.   

installation_id - Randomly generated unique identifier grouping game sessions within a single installed application instance.   

event_count - Incremental counter of events within a game session (offset at 1). Extracted from event_data.   

event_code - Identifier of the event 'class'. Unique per game, but may be duplicated across games. E.g. event code '2000' always identifies the 'Start Game' event for all games. Extracted from event_data.   

game_time - Time in milliseconds since the start of the game session. Extracted from event_data.   

title - Title of the game or video.   

type - Media type of the game or video. Possible values are: 'Game', 'Assessment', 'Activity', 'Clip'.   

world - The section of the application the game or video belongs to. Helpful to identify the educational curriculum goals of the media. Possible values are: 'NONE' (at the app's start screen), TREETOPCITY' (Length/Height), 'MAGMAPEAK' (Capacity/Displacement), 'CRYSTALCAVES' (Weight).   



## Specs Data Description   
This file gives the specification of the various event types.   

event_id - Global unique identifier for the event type. Joins to event_id column in events table.   

info - Description of the event.   

args - JSON formatted string of event arguments. Each argument contains:   
        name - Argument name.   
        type - Type of the argument (string, int, number, object, array).   
        info - Description of the argument.      

#Submission File
For each installation_id represented in the test set, you must predict the accuracy_group of the last assessment for that installation_id. The files must have a header and should look like the following:

installation_id,accuracy_group
00abaee7,3
01242218,0
etc.
