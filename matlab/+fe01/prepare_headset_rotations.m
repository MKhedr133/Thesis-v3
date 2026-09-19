function result = prepare_headset_rotations(tracker)
%PREPARE_HEADSET_ROTATIONS Normalize recorded headset quaternion samples.
result = fe01.fe01_internal("prepare_headset_rotations", tracker);
end
