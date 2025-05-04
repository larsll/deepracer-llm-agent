/**
 * Log levels for controlling output verbosity
 */
export enum LogLevel {
  NONE = 0,
  ERROR = 1,
  WARN = 2,
  INFO = 3,
  DEBUG = 4,
}

/**
 * Custom logger with level filtering
 */
export class Logger {
  private level: LogLevel;
  private name: string;

  /**
   * Create a new logger
   * @param name Name/category for this logger
   * @param level Initial log level
   */
  constructor(name: string = "", level: LogLevel = LogLevel.INFO) {
    this.name = name;
    this.level = this.getLogLevelFromEnv() || level;
  }

  /**
   * Get log level from environment variable
   */
  private getLogLevelFromEnv(): LogLevel | undefined {
    const envLevel = process.env.LOG_LEVEL?.toUpperCase();
    if (!envLevel) return undefined;

    switch (envLevel) {
      case "NONE":
        return LogLevel.NONE;
      case "ERROR":
        return LogLevel.ERROR;
      case "WARN":
        return LogLevel.WARN;
      case "INFO":
        return LogLevel.INFO;
      case "DEBUG":
        return LogLevel.DEBUG;
      default:
        return undefined;
    }
  }

  /**
   * Set the current log level
   */
  setLevel(level: LogLevel): void {
    this.level = level;
  }

  /**
   * Format a log message with optional timestamp and logger name
   */
  private formatMessage(message: string): string {
    // const prefix = this.name ? `[${this.name}] ` : '';
    return message;
  }

  /**
   * Log an error message
   */
  error(message: string, ...args: any[]): void {
    if (this.level >= LogLevel.ERROR) {
      console.error(`❌ ${this.formatMessage(message)}`, ...args);
    }
  }

  /**
   * Log a warning message
   */
  warn(message: string, ...args: any[]): void {
    if (this.level >= LogLevel.WARN) {
      console.warn(`⚠️ ${this.formatMessage(message)}`, ...args);
    }
  }

  /**
   * Log an informational message
   */
  info(message: string, ...args: any[]): void {
    if (this.level >= LogLevel.INFO) {
      console.log(this.formatMessage(message), ...args);
    }
  }

  /**
   * Log a debug message
   */
  debug(message: string, ...args: any[]): void {
    if (this.level >= LogLevel.DEBUG) {
      console.log(`🔍 ${this.formatMessage(message)}`, ...args);
    }
  }
}

/**
 * Get a logger instance with the specified name
 * @param name Name for the logger (typically module/class name)
 * @param level Initial log level
 */
export function getLogger(name: string = "", level?: LogLevel): Logger {
  return new Logger(name, level);
}
