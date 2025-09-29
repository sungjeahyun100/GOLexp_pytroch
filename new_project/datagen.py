# -*- coding: utf-8 -*-
import ctypes as ct
import os
import argparse
import sys

def find_library():
    """공유 라이브러리를 여러 경로에서 찾기"""
    possible_paths = [
        '../build/GOLdatagen.so',
        './build/GOLdatagen.so', 
        'build/GOLdatagen.so',
        os.path.join(os.path.dirname(__file__), '..', 'build', 'GOLdatagen.so')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    print("❌ 공유 라이브러리를 찾을 수 없습니다!")
    print("다음 경로들을 확인했습니다:")
    for path in possible_paths:
        print("  - " + os.path.abspath(path))
    print("\n해결 방법:")
    print("1. 프로젝트 루트에서 'mkdir build && cd build && cmake .. && make' 실행")
    print("2. CUDA가 없는 경우 --cpu 옵션 사용")
    return None

# 공유 라이브러리 로드 (안전한 방식)
lib_path = find_library()
if lib_path is None:
    print("⚠️  GPU 라이브러리를 사용할 수 없습니다. --cpu 옵션을 사용하세요.")
    lib = None
else:
    try:
        lib = ct.CDLL(lib_path)
        print("✅ 라이브러리 로드 성공: " + lib_path)
    except Exception as e:
        print("❌ 라이브러리 로드 실패: " + str(e))
        lib = None

# 함수 시그니처 설정 (라이브러리가 로드된 경우만)
if lib is not None:
    lib.genGOLdata.argtypes = [ct.c_uint32, ct.c_uint32, ct.c_double]
    lib.genGOLdata.restype = None

    lib.genGOLdataInOneFile.argtypes = [ct.c_uint32, ct.c_uint32, ct.c_double]
    lib.genGOLdataInOneFile.restype = None

    lib.genGOLdataInHost.argtypes = [ct.c_uint32, ct.c_uint32, ct.c_double]
    lib.genGOLdataInHost.restype = None

def main():
    parser = argparse.ArgumentParser(description='Game of Life 데이터 생성기')
    parser.add_argument('param1', type=int, help='첫 번째 매개변수 (uint32)')
    parser.add_argument('param2', type=int, help='두 번째 매개변수 (uint32)')
    parser.add_argument('param3', type=float, help='세 번째 매개변수 (double)')
    
    # 선택적 인자들
    parser.add_argument('--verbose', '-v', action='store_true', help='상세 출력')
    parser.add_argument('--output', '-o', type=str, help='출력 디렉토리')
    parser.add_argument('--one_file', action='store_true', help='한 파일에 데이터를 몰아서 저장')
    parser.add_argument('--cpu', action='store_true', help="외장 글카 없는 경우에 cpu로 데이터 생성이 가능케 함")
    
    args = parser.parse_args()
    
    if args.verbose:
        print("매개변수: {}, {}, {}".format(args.param1, args.param2, args.param3))
        if args.output:
            print("출력 디렉토리: " + args.output)
    
    # 라이브러리 로드 확인
    if lib is None:
        print("❌ C++ 라이브러리를 사용할 수 없습니다!")
        print("해결 방법:")
        print("1. 프로젝트 빌드: mkdir build && cd build && cmake .. && make")
        print("2. 또는 Docker 사용: docker-compose up golexp-cpu")
        return 1
    
    # C 라이브러리 함수 호출
    if args.cpu:
        try:
            print("🔧 CPU 모드로 데이터 생성 중...")
            lib.genGOLdataInHost(args.param1, args.param2, args.param3)
            print("✅ CPU 데이터 생성 완료!")
        except Exception as e:
            print("❌ CPU 모드 오류: " + str(e))
            return 1
    else:
        if args.one_file:
           try:
               print("📁 단일 파일 모드로 데이터 생성 중...")
               lib.genGOLdataInOneFile(args.param1, args.param2, args.param3)
               print("✅ 단일 파일 데이터 생성 완료!")
           except Exception as e:
               print("❌ 단일 파일 모드 오류: " + str(e))
               return 1
        else:
           try:
               print("🚀 GPU 모드로 데이터 생성 중...")
               lib.genGOLdata(args.param1, args.param2, args.param3)
               print("✅ GPU 데이터 생성 완료!")
           except Exception as e:
               print("❌ GPU 모드 오류: " + str(e))
               return 1
        
        return 0

if __name__ == "__main__":
    sys.exit(main())
