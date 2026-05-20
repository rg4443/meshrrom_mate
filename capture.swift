import Foundation
import RealityKit

@main
struct ObjectCaptureCLI {
    static func main() async throws {
        let arguments = CommandLine.arguments
        if arguments.count < 3 {
            print("Usage: capture_tool <input-folder> <output-file.usdz>")
            exit(1)
        }
        
        let inputFolder = URL(fileURLWithPath: arguments[1])
        let outputFile = URL(fileURLWithPath: arguments[2])
        
        // Ensure the Mac supports hardware Object Capture
        guard PhotogrammetrySession.isSupported else {
            print("[ERROR] Object Capture is not supported on this Mac hardware.")
            exit(1)
        }
        
        print("[Native Capture] Initializing Apple Object Capture session...")
        let session = try PhotogrammetrySession(input: inputFolder)
        
        print("[Native Capture] Processing photos (medium quality)...")
        try session.process(requests: [.modelFile(url: outputFile, detail: .medium)])
        
        // Fixed: Added 'try' here because session.outputs can throw errors
        for try await output in session.outputs {
            switch output {
            case .requestProgress(_, let fraction):
                let percentage = Int(fraction * 100)
                print("[Native Capture] Progress: \(percentage)%")
            case .requestComplete(_, _):
                print("[Native Capture] Model generation complete!")
                exit(0)
            case .requestError(_, let error):
                print("[Native Capture] Error during processing: \(error)")
                exit(1)
            default:
                break
            }
        }
    }
}