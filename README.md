# Eye Blink Communication System

A computer vision-based communication system designed for paralyzed individuals to communicate using eye blinks. The system uses OpenCV to detect eye blinks and navigate through predefined communication categories and actions.

## Features

- **Eye Blink Detection**: Uses OpenCV's Haar cascade classifiers for face and eye detection
- **Two-Level Navigation**: Navigate through categories and then specific actions
- **Audio Feedback**: Plays beep sounds when selections are made
- **Visual Interface**: Clear on-screen guidance and progress indicators
- **Demo Mode**: Test navigation without a webcam

## How It Works

### Blink Types:
- **Short Blink**: Navigate to the next option
- **Long Blink (1 second)**: Select the current option

### Navigation Flow:
1. **Category Selection**: Navigate through main categories (Personal Needs, Emotions & Comfort, etc.)
2. **Action Selection**: After selecting a category, navigate through specific actions
3. **Confirmation**: System plays a beep and prints the selected action
4. **Reset**: Automatically returns to category selection for next communication

## Installation

1. Ensure you have Python 3.7+ installed
2. Install required packages:
   ```bash
   pip install opencv-python numpy
   ```
3. Make sure you have the `beep.mp3` audio file in the project directory
4. Ensure `data.json` contains the communication data

## Usage

### Normal Mode (with webcam):
```bash
python main.py
```

### Demo Mode (without webcam):
```bash
python main.py --demo
```

In demo mode, use these keys:
- `s`: Simulate short blink (navigate)
- `l`: Simulate long blink (select)
- `b`: Go back to categories
- `r`: Reset to beginning
- `q`: Quit

## Controls

- **Short Blink / 's' key**: Navigate through options
- **Long Blink / 'l' key**: Select current option
- **'b' key**: Go back from actions to categories
- **'r' key**: Reset to beginning
- **'q' key**: Quit the application

## System Requirements

- **Camera**: Any USB webcam or built-in camera
- **Audio**: System capable of playing MP3 files
- **macOS**: Uses `afplay` for audio (can be modified for other systems)
- **Lighting**: Good lighting conditions for reliable face/eye detection

## Configuration

### Timing Settings:
- `LONG_BLINK_THRESHOLD`: 1.0 second (time to hold blink for selection)
- `blink_cooldown`: 0.5 seconds (minimum time between navigation blinks)
- `EYE_CLOSED_THRESHOLD`: 3 frames (frames to confirm eyes are closed)
- `EYE_OPEN_THRESHOLD`: 2 frames (frames to confirm eyes are open)

### Camera Settings:
- Resolution: 640x480 (optimized for performance)
- Frame rate: Depends on system capabilities

## Communication Data

The system uses `data.json` which contains:
- 5 main categories with 20 actions each
- Categories: Personal Needs, Emotions & Comfort, Social & Communication, Environmental Controls, Health Monitoring & Emergency

## Troubleshooting

### Common Issues:

1. **"Could not open webcam"**:
   - Check if camera is connected and not used by another application
   - Try running in demo mode first: `python main.py --demo`

2. **Poor eye detection**:
   - Ensure good lighting conditions
   - Position face clearly in front of camera
   - Avoid reflections on glasses

3. **Audio not working**:
   - Ensure `beep.mp3` file exists
   - Check system audio settings
   - Verify `afplay` command works (macOS)

4. **False blink detection**:
   - Adjust lighting conditions
   - Modify threshold values in code if needed

## Accessibility Features

- **Large, clear text**: Easy-to-read interface
- **High contrast**: Clear visual indicators
- **Audio feedback**: Confirms selections
- **Progressive disclosure**: Shows only relevant options
- **Reset functionality**: Easy to start over

## Technical Details

### Detection Algorithm:
1. **Face Detection**: Haar cascade classifier
2. **Eye Detection**: Enhanced eye detection with eyeglasses support
3. **Blink State Machine**: Tracks eye open/closed states
4. **Timing Logic**: Differentiates between short and long blinks

### Performance Optimizations:
- Reduced video resolution for faster processing
- Histogram equalization for better detection in varying light
- Optimized cascade parameters
- Background audio playback

## Future Enhancements

- Support for gaze direction detection
- Customizable communication phrases
- Multiple language support
- Voice synthesis for selected phrases
- Calibration mode for personalized thresholds
- Log communication history

## Contributing

This system is designed to help paralyzed individuals communicate more effectively. Contributions that improve accessibility, reliability, or usability are welcome.

## License

This project is intended for assistive technology use and educational purposes.
