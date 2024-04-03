## Imports

import argparse
import sys

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
import platform

import sys
from gi.repository import GObject, Gst
from gi.repository import GObject, Gst, GstRtspServer

def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream\n")
        loop.quit()
    elif t==Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write("Warning: %s: %s\n" % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write("Error: %s: %s\n" % (err, debug))
        loop.quit()
    return True

def is_aarch64():
    return platform.uname()[4] == 'aarch64'

import pyds

GObject.threads_init()
Gst.init(None)
    
pipeline = Gst.Pipeline()
source = Gst.ElementFactory.make("filesrc", "file-source")
if source is None:
    print ("Source plugin is null! ")
    exit()
h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
# Make the UDP sink
updsink_port_num = 5400
sink = Gst.ElementFactory.make("udpsink", "udpsink")
if not sink:
    sys.stderr.write(" Unable to create udpsink")

sink.set_property('host', '224.224.255.255')
sink.set_property('port', updsink_port_num)
sink.set_property('async', False)
sink.set_property('sync', 1)
stream_path = '/opt/nvidia/deepstream/deepstream/samples/streams/' + "sample_720p.h264" 

print("Playing file %s " %stream_path)
source.set_property('location', stream_path)
streammux.set_property('width', 1920)
streammux.set_property('height', 1080)
streammux.set_property('batch-size', 1)
streammux.set_property('batched-push-timeout', 4000000)
bitrate = 4000000
encoder.set_property('bitrate', bitrate)
if is_aarch64():
    encoder.set_property('preset-level', 1)
    encoder.set_property('insert-sps-pps', 1)
    encoder.set_property('bufapi-version', 1)
        
pipeline.add(source)
pipeline.add(h264parser)
pipeline.add(decoder)
pipeline.add(streammux)
pipeline.add(encoder)
pipeline.add(rtppay)
pipeline.add(sink)

source.link(h264parser)
h264parser.link(decoder)

sinkpad = streammux.get_request_pad("sink_0")
if not sinkpad:
    sys.stderr.write(" Unable to get the sink pad of streammux \n")

srcpad = decoder.get_static_pad("src")
if not srcpad:
    sys.stderr.write(" Unable to get source pad of decoder \n")

srcpad.link(sinkpad)

streammux.link(encoder)
encoder.link(rtppay)
rtppay.link(sink)

# create an event loop and feed gstreamer bus mesages to it
loop = GObject.MainLoop()
bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect ("message", bus_call, loop)

# Start streaming
rtsp_port_num = 8554
codec = "H264"
server = GstRtspServer.RTSPServer.new()
server.props.service = "%d" % rtsp_port_num
server.attach(None)

factory = GstRtspServer.RTSPMediaFactory.new()
factory.set_launch( "( udpsrc name=pay0 port=%d buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 \" )" % (updsink_port_num, codec))
factory.set_shared(True)
server.get_mount_points().add_factory("/ds-test", factory)

print("\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d/ds-test ***\n\n" % rtsp_port_num)
# start play back and listen to events
print("Starting pipeline \n")
pipeline.set_state(Gst.State.PLAYING)
try:
    loop.run()
except:
    pass
# cleanup
pipeline.set_state(Gst.State.NULL)