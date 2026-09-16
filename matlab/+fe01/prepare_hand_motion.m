function series = prepare_hand_motion(tracker, hand)
%PREPARE_HAND_MOTION Prepare recorded 3-D positions, distances, and speeds.
series = fe01.fe01_internal("prepare_hand_motion", tracker, hand);
end
