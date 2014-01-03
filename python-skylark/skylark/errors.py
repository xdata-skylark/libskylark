class SkylarkError(Exception):
    pass

class DimensionMistmatchError(SkylarkError):
    pass

class UnexpectedLowerLayerError(SkylarkError):
    pass

class UnsupportedError(SkylarkError):
    pass

class InvalidObjectError(SkylarkError):
    pass
