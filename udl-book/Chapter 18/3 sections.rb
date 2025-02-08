3 sections 
- Mandatory 
- Service/Usage personalization
- Reco 

"100" -> 
"110" -> UAV store
"101" -> Reco
"111" -> UAV store, Reco

100,111,101,110
(legitimate interest)
- Client sends all the events below.
bookmarks
viewing history
playbackPause,
playerError
genericError
playbackSkipAhead,
playbackSkipBack,
playbackHeartbeat 
playbackmetrics
privacyPolicy
watch
scheduleRecording
downloadTriggered

110, 111
(client send the request)
(UAV will store them as configured)
thirdPartyAppStart
returnToLauncher
deepLinkTriggered
appStart
appEnd
adWatched
adDelivered
adSkipped
railView
railSelection

101, 111
Fwd to reco engine
(pre-defined list of events.)
(regular operation.)