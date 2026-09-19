function result = read_participant_hand(path)
%READ_PARTICIPANT_HAND Read the reported left/right hand without guessing.
result = fe01.fe01_internal("read_participant_hand", path);
end
